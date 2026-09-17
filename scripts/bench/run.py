"""量化门槛的执行器（D-13：LoCoMo ≥ 70 / LongMemEval ≥ 75）。

## 三种模式，回答三个不同的问题

| 模式 | 需要什么 | 回答什么 |
|---|---|---|
| `--inspect` | 零 | **加载器读出了什么**（先看这个，再信任何分数） |
| `--answer-mode extract` | **零模型** | 机制（提取 / 召回 / 时态 / 去重）对不对——**可进 CI** |
| `--answer-mode llm` | 真实网关 | 端到端（含生成）能到多少分 |

分开的理由：如果一个数字同时依赖"检索对不对"和"生成好不好"，
它掉下去时你**不知道该修哪一半**。`extract` 模式把生成这一半拿掉，
剩下的全部归因于机制——这正是 G6 要的"证明机制有效"。

## 用法

    python scripts/bench/run.py --inspect
    python scripts/bench/run.py                                  # 自建集 + extract
    python scripts/bench/run.py --dataset locomo --path locomo10.json
    REALTEST_API_KEY=... python scripts/bench/run.py --answer-mode llm

退出码：达到门槛 / 无退步 → 0；否则 1（可直接当 CI 门禁用）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# 加 **`scripts/`** 而不是 `scripts/bench/`：`bench` 是个命名空间包，
# 要 `import bench.datasets` 就得让 Python 看到它的**父目录**。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.datasets import DatasetError, Question, load
from bench.metrics import Metrics

CASES_DIR = Path(__file__).resolve().parent / "cases"
DATA_DIR = Path(__file__).resolve().parent / "data"

_DEFAULT_PATHS = {
    "selfcheck": CASES_DIR / "selfcheck.jsonl",
    "locomo": DATA_DIR / "locomo10.json",
    "longmemeval": DATA_DIR / "longmemeval_oracle.json",
}


def _resolve_path(dataset: str, given: str | None) -> Path:
    """决定这次要读哪个文件。

    **默认路径必须按数据集分派，不能只有一个默认值。**
    只写自建集那一个默认值时，`--dataset locomo` 且忘记 `--path` 会**静默去读自建集**，
    然后在 JSON 解析那层炸开——而那条报错**完全看不出"是文件选错了"**。

    （这不是假想的：**本函数就是因为踩了这个坑才写出来的**。）
    """
    if given:
        return Path(given)
    resolved = _DEFAULT_PATHS.get(dataset)
    if resolved is None:
        raise SystemExit(f"未知数据集 {dataset!r}；支持 {sorted(_DEFAULT_PATHS)}")
    if not resolved.exists():
        raise SystemExit(
            f"默认路径不存在：{resolved}\n"
            f"先取数据集：python scripts/bench/fetch.py {dataset}\n"
            f"或显式指定：--path <文件>"
        )
    return resolved

THRESHOLDS = {"locomo": 70.0, "longmemeval": 75.0}
"""D-13 的量化门槛。**只有对外标尺有绝对门槛**——

自建集不设绝对线：它的用途是"向内自证机制没坏"，判据是**相对基线不退步**（`--baseline`）。
给自建集定一个绝对分数，会诱导人去调题而不是修系统。
"""

_API_KEY_ENV = "ARTIFACT_SPIRIT_API_KEY"


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #


def _bench_home(root: Path, name: str, *, with_llm: bool, with_embedding: bool) -> Path:
    """造一个干净的 hermes_home。**每题一个库**，题与题之间不互相污染。"""
    from artifact_spirit.config import save

    home = root / _safe(name)
    home.mkdir(parents=True, exist_ok=True)
    values: dict[str, object] = {
        "spirit.name": "评测器灵",
        "backend.path": str(home / "spirit" / "spirit.db"),
    }
    if with_llm:
        values.update(
            {
                "models.llm.provider": "openai-compat",
                "models.llm.base_url": _base_url(),
                "models.llm.api_key_env": _API_KEY_ENV,
                # **档位要显式配**：不配的话会落到 `DEFAULT_LLM_TASK_MODELS` 的默认名
                # （`glm-5.3-flash` 等），于是"换个模型试试"这件事根本发生不了——
                # 你会以为换了，实际跑的还是默认那个。
                "models.llm.extract": _model("REALTEST_MODEL_SMALL", "glm-5.3-flash"),
                "models.llm.dedup": _model("REALTEST_MODEL_SMALL", "glm-5.3-flash"),
                "models.llm.summarize": _model("REALTEST_MODEL_SMALL", "glm-5.3-flash"),
                "models.llm.consolidate": _model("REALTEST_MODEL_LARGE", "glm-5.3"),
                "models.llm.soul": _model("REALTEST_MODEL_LARGE", "glm-5.3"),
                "models.llm.crosscheck": _model("REALTEST_MODEL_SMALL", "glm-5.3-flash"),
            }
        )
    if with_embedding:
        values.update(
            {
                "models.embedding.provider": "openai-compat",
                "models.embedding.base_url": _base_url(),
                "models.embedding.model": _model(
                    "REALTEST_EMBED_MODEL", "kinfra-text-embedding-4b"
                ),
                "models.embedding.api_key_env": _API_KEY_ENV,
                "models.embedding.dim": int(_model("REALTEST_EMBED_DIM", "2560")),
            }
        )
    save(home, values)
    return home


@contextlib.contextmanager
def _workdir(fixed: str | None):
    """评测的工作目录。

    给了 `--home` 就**不删**——于是下一轮能复用已经灌好的库：
    换口径、换作答模型重跑时不必再花十几分钟提取，
    而且两次运行落在**同一个记忆库**上（那才可比）。
    这也更接近真实形态：记忆库是**持续积累**的，不是每次提问都重建。
    """
    if fixed:
        path = Path(fixed)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="aspirit-bench-") as tmp:
        yield Path(tmp)


def _db_file(home: Path) -> Path:
    return home / "spirit" / "spirit.db"


def _safe(name: str) -> str:
    """题目 id → 文件名。**非 ASCII 与非字母数字一律替换**——
    Windows 上带 `#`/`:` 的目录名会直接创建失败，而失败信息很难看懂是评测脚本的问题。"""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]


def _env(*, need_key: bool) -> dict[str, str]:
    """交给器灵的环境变量视图。`need_key` 为假时**什么都不给**（不是给个空串）。

    取名 `need_key` 而不是 `with_llm`，是因为**嵌入也要密钥**——
    早先按"用不用 LLM"来决定注入，于是 `--embedding on` 在 extract 模式下
    拿到的是**没有密钥的环境**：嵌入静默失败、召回降级成关键词，
    而 R@k 照样打得出来，只是它衡量的不再是语义召回（实测 0.7 秒跑完，暴露了这一点）。
    """
    key = os.environ.get("REALTEST_API_KEY", "")
    return {_API_KEY_ENV: key} if (need_key and key) else {}


def _model(env_name: str, default: str) -> str:
    """从环境变量取模型名，缺省给一个**真实存在于网关**的默认。

    模型名写错时网关会报 404，而 404 长得像"端点不对"——
    评测脚本报"模型不存在"比报"网关不通"省一次排查。
    """
    return os.environ.get(env_name) or default


def _base_url() -> str:
    return os.environ.get("REALTEST_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")


@lru_cache(maxsize=1)
def _gen_client():
    """评测侧的 HTTP 客户端。**复用同一个连接池**——

    每次调用新建一个 client 会把"评测一个数据集"变成"开几千条 TCP"，
    而那既是时间开销，也可能撞上网关的连接数限制（表现为随机失败，
    看起来像"模型不稳定"）。
    """
    import httpx

    return httpx.Client(base_url=_base_url(), timeout=httpx.Timeout(60.0, connect=10.0))


def _llm_generate(system: str, user: str) -> str | None:
    """评测侧的**生成**调用。失败返回 `None`——评测不该因一次网关抖动整体崩掉。

    ## 为什么不用器灵的 `core.llm`

    因为那会让"用 LLM 灌入"变成**隐式强制的**：`core.llm` 一存在，
    `ingest_turn` 的提取器就会调它，而**提取质量是另一个变量**。

    评测要能分别回答两个问题：

    - "召回 + 生成行不行"（灌入用规则提取，秒级、零成本、可反复跑）；
    - "LLM 提取行不行"（灌入也用模型，贵 100 倍）。

    混在一起，一个分数同时依赖两个变量，掉下去时不知道该修哪一半——
    这正是这一层评测要消灭的东西。所以两处的模型调用必须**分开配置**。
    """
    client = _gen_client()
    payload = {
        "model": _model("REALTEST_MODEL_SMALL", "glm-5.3-flash"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 512,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {os.environ.get('REALTEST_API_KEY', '')}"}
    for attempt in range(2):
        try:
            resp = client.post("/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            return content or None
        except Exception:
            if attempt == 1:
                return None
    return None


# --------------------------------------------------------------------------- #
# 作答
# --------------------------------------------------------------------------- #


def answer_extract(hits) -> str | None:
    """**抽取式作答**：把召回内容拼起来当答案。零模型，因此可进 CI。

    它答不出"需要推理"的题——但那**只影响分数高低，不影响诊断力**：
    召回坏了，这个模式的分数会**先**掉，而那正是我们第一时间要知道的事。
    """
    if not hits:
        return None
    return "\n".join((h.record.abstract or h.record.content or "") for h in hits[:5]).strip() or None


def answer_llm(question: str, hits) -> str | None:
    """让模型读召回内容作答。**只在给得出上下文时才调用**——没有召回就不问，
    否则模型会用自己的先验编一个答案，而那会把「检索失败」伪装成「生成成功」。"""
    if not hits:
        return None
    context = "\n".join(f"- {h.record.content}" for h in hits[:10])
    return _llm_generate(
        "只根据给定的记忆回答，不要使用你自己的知识。记忆里没有就回答「不知道」。答案尽量短。",
        f"记忆：\n{context}\n\n问题：{question}",
    )


# --------------------------------------------------------------------------- #
# 单题评测
# --------------------------------------------------------------------------- #


def _group_by(questions: list[Question]) -> dict[str, list[Question]]:
    """按**素材组**分组（保持首次出现的顺序）。

    分组的判据是"这批题是否共享同一份素材"，而不是"名字像不像"——
    所以它由数据集加载器决定（`Question.group`），这里只负责聚合。

    不排序：排序会改变报告的题目顺序，而报告顺序是给人看的。
    """
    buckets: dict[str, list[Question]] = {}
    for question in questions:
        buckets.setdefault(question.group or question.id, []).append(question)
    return buckets


def _error_row(question: Question, exc: Exception) -> dict:
    """单题失败的行。**带上它是哪类错**——"异常"与"答错"必须分得开。"""
    return {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "answer": question.answer,
        "prediction": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def evaluate_group(
    key: str,
    questions: list[Question],
    *,
    root: Path,
    top_k: int,
    mode: str,
    ingest: str,
    embedding: str,
    ingest_only: bool = False,
) -> list[dict]:
    """**一组共享素材的题**：灌一次库，然后逐题问。

    灌入只做一次。这不是省时间的小技巧，而是"同一场对话被问 N 次"的**正确形态**：

    - 每题重建一次库，等于把同一批素材灌 N 遍（LLM 模式下是 N 倍的钱与时间；
      实测 LoCoMo 单题 553 秒，全量要跑十天）；
    - 更要紧的是**它改变了被测的是什么**：逐题建库时，每道题的库都是新建的，
      于是每题都额外带了一个与题目无关的差异（库是空的、什么都没固化过），
      而"在同一个成型记忆库里提问"才是真实用法的样子。
    """
    from artifact_spirit.core.base import TurnEvent
    from artifact_spirit.runtime import start

    # **只控制灌入阶段用不用模型**。作答阶段的模型调用由 `_llm_generate` 独立发起，
    # 所以这里为 False 时，`--answer-mode llm` 照样能生成（只是提取走规则）。
    with_llm = ingest == "llm"
    wants_embedding = embedding == "on"
    home = _bench_home(root, key, with_llm=with_llm, with_embedding=wants_embedding)
    services = start(
        str(home), env=_env(need_key=with_llm or wants_embedding), start_threads=False
    )
    # **确认嵌入真的可用**。不可用时召回会**静默**降级成关键词检索，
    # 而 R@k 照样打得出来——只是它衡量的不再是语义召回。
    # 这类"指标还在、含义已经换了"的降级，比直接报错危险得多。
    if wants_embedding and not services.resolver.embedding_available():
        print(
            "  ⚠ 本轮指定了 --embedding on，但器灵判定嵌入不可用——"
            "R@k 实际是**纯关键词检索**的水平，不要当成语义召回能力来读"
        )
    # **判据是「库里有没有东西」，不是「文件在不在」**：
    # 库文件在 `start()` 建表时就被创建了（空的、几 KB），
    # 所以 `exists()` 在**第一次跑**就为真——灌入被跳过、库永远是空的，
    # 而报告只会说「答题率 0%」，把原因指向检索。
    installed = len(services.backend.query(status=None))
    db = _db_file(home)
    rows: list[dict] = []
    try:
        if installed:
            print(
                f"  [复用] {key}：库里已有 {installed} 条记忆"
                f"（{db.stat().st_size // 1024} KB），跳过灌入"
                "——换口径或换作答模型重跑时不必再提取一遍"
            )
        else:
            for index, turn in enumerate(questions[0].turns):
                intents = services.core.ingest_turn(
                    TurnEvent(
                        session_id=turn.session_id,
                        user=turn.text,
                        assistant="",
                        ts=turn.ts or f"2026-01-{index + 1:02d}T10:00:00+08:00",
                    )
                )
                services.write_now(intents)

        if ingest_only:
            print(f"  [灌入完成] {key}：{len(questions[0].turns)} 段素材已入库")
            return []

        for question in questions:
            try:
                rows.append(_ask(services, question, top_k=top_k, mode=mode))
            except Exception as exc:
                rows.append(_error_row(question, exc))
    finally:
        services.stop(drain=True)
    return rows


def _ask(services, question: Question, *, top_k: int, mode: str) -> dict:
    """在**已经灌好**的库上问一题。"""
    from artifact_spirit.core.base import RecallQuery

    hits = services.core.recall(RecallQuery(text=question.question, top_k=top_k))
    pred = answer_llm(question.question, hits) if mode == "llm" else answer_extract(hits)
    retrieved = [h.record.id for h in hits]
    gold_ids = _gold_ids(services, question)
    return {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "answer": question.answer,
        "prediction": pred,
        # **两个 id 列表都要带出来**：判分与"分类别报告"都依赖它们。
        # 它们必须来自**同一次召回**——重新查一遍会拿到另一个 top_k 顺序，
        # 于是"总分"与"分类别分"来自两次不同的运行，对不上账。
        "retrieved_ids": retrieved,
        "gold_ids": sorted(gold_ids),
        "hit": bool(set(retrieved[:top_k]) & gold_ids) if gold_ids else False,
    }


def _readable_ts(value: str | None) -> str:
    """ISO8601 → `8 May 2023` 这类**人可读**形式（供关键词匹配）。

    手工拼 day / month / year，不用 `strftime("%-d")`：
    后者是 glibc 扩展，Windows 上不可用——而评测要在哪台机器上都给出同一个字符串。
    """
    if not value:
        return ""
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return ""
    return f"{stamp.day} {stamp.strftime('%B')} {stamp.year}"


def _searchable_text(record) -> str:
    """一条记忆的**可搜索文本**：判定"它算不算这题的相关记忆"时拿什么去找它。

    为什么要带上时间的**人可读形式**——这是被数据逼出来的：

    LoCoMo 的时间类题，标准答案就是「7 May 2023」这种日期，
    而那个日期在这条记忆里**只存在于 `valid_from`**：
    LLM 把对话改写成「Caroline went to the LGBTQ support group」之后，
    日期就不在 `content` 里了。

    只扫 `content` 的后果不是"少算几条"，而是**整类题恒判为没召回**：
    temporal 占 LoCoMo 的 321/1540，它的 R@k 会系统性地压在低处，
    于是「检索不行」这个结论里混进了一个**纯口径的假象**。
    """
    parts = [
        record.content or "",
        record.abstract or "",
        record.object or "",
        _readable_ts(record.valid_from),
    ]
    return " ".join(p for p in parts if p).casefold()


def _matches_gold(text: str, keywords: list[str]) -> bool:
    """这段文本命中了足够多的关键词吗——**足够多是几个**，见 `_gold_ids`。"""
    if not keywords:
        return False
    needed = min(2, len(keywords))
    return sum(1 for k in keywords if k in text) >= needed


def _gold_ids(services, question: Question) -> set[str]:
    """**从全库**找相关内容，而不是从召回结果里找。

    这个区别决定了检索指标的分母：
    拿召回结果当全集，等于假设"召回一定是对的"——检索全漏时指标会显示 0 相关记忆，
    于是 recall 平凡地等于 1，**指标在系统最坏的时候最好看**。

    ## 判据：命中**几个**关键词才算相关

    用 `any`（命中一个就算）会**虚高**：时间类题的关键词里总有年份
    （`['May', '2023']`），而"2023"几乎每条当年记忆都含——
    一个词命中全库，"相关记忆"于是等于整个库，recall 平庸地接近 1。

    所以要求命中 `min(2, 关键词数)` 个：

    - `['May', '2023']` → 需**同时**含 `May` 与 `2023`，只剩 5 月那几条 ✓ 有区分度
    - `['2022']` → 只有一词，命中它即可 ✓
    - `['The', 'sunday', 'before', '25', 'May', '2023']` → 虚词无害，
      `May` + `2023` 已足够把它挑出来 ✓
    """
    wanted = [k.casefold() for k in question.gold_keywords]
    if not wanted:
        return set()
    # **有官方标注就用官方标注**，关键词近似只是没有标注时的退路。
    wanted_sessions = set(question.evidence_sessions)
    return {
        record.id
        for record in services.backend.query(status=None)
        if (
            record.source_session in wanted_sessions
            if wanted_sessions
            else _matches_gold(_searchable_text(record), wanted)
        )
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    path = _resolve_path(args.dataset, args.path)
    try:
        questions = load(args.dataset, path)
    except DatasetError as exc:
        print(f"[加载失败] {exc}", file=sys.stderr)
        return 2

    print(f"数据集：{args.dataset}（{path}）")
    print(f"题数：{len(questions)}；模式：{args.answer_mode}；top_k：{args.top_k}")

    if args.inspect:
        return _inspect(questions)

    if args.limit:
        questions = questions[: args.limit]
        print(f"[--limit] 只跑前 {len(questions)} 题（先验链路，再跑全量）")

    thresholds = _thresholds(questions, args)
    if thresholds is None:
        return 2

    with _workdir(args.home) as root:
        metrics = Metrics()
        details = []
        groups = _group_by(questions)
        if len(groups) < len(questions):
            print(
                f"[分组] {len(questions)} 题归为 {len(groups)} 组，每组只灌一次库"
                f"——省下 {len(questions) - len(groups)} 次重复灌入"
            )
        for key, group in groups.items():
            try:
                rows = evaluate_group(
                    key,
                    group,
                    root=root,
                    top_k=args.top_k,
                    mode=args.answer_mode,
                    ingest=args.ingest,
                    embedding=args.embedding,
                    ingest_only=args.ingest_only,
                )
            except Exception as exc:
                # 一组崩掉不该掩掉其余组的结果——但**同一组的题一起失败**，
                # 因为它们的库是同一个：库灌不进去，这一组就都没有可问的东西了。
                rows = [_error_row(question, exc) for question in group]

            if args.ingest_only:
                continue

            # `strict=True`：行数与题数不等说明分组逻辑错了，那会让分数与题目**错位**——
            # 而错位的分数比没有分数更坏。
            for question, row in zip(group, rows, strict=True):
                details.append(row)
                metrics.add(
                    pred=row.get("prediction"),
                    gold=question.answer,
                    retrieved=_ids_of(row),
                    gold_ids=set(row.get("gold_ids") or ()),
                    k=args.top_k,
                )
                _track_category(metrics, question.category, row, k=args.top_k)
                if args.verbose:
                    mark = "OK " if row.get("prediction") else "-- "
                    print(f"  [{mark}] {question.id}: {str(row.get('prediction'))[:60]!r}")

    if args.ingest_only:
        print(
            "\n只灌入模式：库已就绪。之后用 `--home` 指向同一目录即可复用它评测——"
            "\n换口径、换作答模型都不必再提取一遍。"
        )
        return 0

    report = metrics.to_dict()
    _print_report(report, details, args)
    return _verdict(report, thresholds, args, details)


def _ids_of(row: dict) -> list[str]:
    return list(row.get("retrieved_ids") or ())


def _track_category(metrics: Metrics, category: str, row: dict, *, k: int) -> None:
    """按类别单独累计——**总分相同的两种系统，下一步可能完全相反**。"""
    bucket = metrics.by_category.setdefault(category, Metrics())
    bucket.add(
        pred=row.get("prediction"),
        gold=row.get("answer") or "",
        retrieved=_ids_of(row),
        gold_ids=set(row.get("gold_ids") or ()),
        k=k,
    )


def _thresholds(
    questions: list[Question], args: argparse.Namespace
) -> list[tuple[str, float]] | None:
    """决定这次跑分要判定哪些门槛。

    **公开标尺的绝对门槛只在 `--answer-mode llm` 下判定。**
    `extract` 模式把召回内容整段当答案，分数天然远低于公开标尺的口径——
    拿它去比「LoCoMo ≥ 70」只会得到一个"系统很差"的**错误结论**，
    而那个结论会把人引向完全错误的优化方向。
    """
    if args.dataset in THRESHOLDS:
        if args.answer_mode != "llm":
            print(
                f"\n[提示] {args.dataset} 的绝对门槛（≥ {THRESHOLDS[args.dataset]}）"
                "**只在 `--answer-mode llm` 下才有意义**。"
            )
            print("       本次是 extract 模式——分数不可与公开标尺比较，**不判定门槛**。")
            return []
        return [(f"{args.dataset} ≥ {THRESHOLDS[args.dataset]}", THRESHOLDS[args.dataset])]
    if args.min_accuracy is not None:
        return [(f"自建集 ≥ {args.min_accuracy}", float(args.min_accuracy))]
    print("\n[提示] 自建集没有绝对门槛——用 --min-accuracy 显式给一个，")
    print("       或把分数存成基线后用 --baseline 比。这样设计是刻意的：")
    print("       给自建集定死一个线，会诱导人去调题，而不是去修系统。")
    return []


def _verdict(report: dict, thresholds: list[tuple[str, float]], args, details: list[dict]) -> int:
    failures: list[str] = []
    score = report["accuracy"] * 100

    # **异常优先于门槛判定**，且本身就算失败。
    # 一个"每题都抛异常"的运行会得到 accuracy=0——如果只看分数，
    # 它在没有绝对门槛时会**判定通过**：门禁报绿，而评测其实什么都没测到。
    # 这与 P0-12 的"把丢写说成干净收尾"是同一类错误——**在最需要报案的时刻报平安**。
    errors = [row for row in details if row.get("error")]
    if errors:
        failures.append(
            f"{len(errors)}/{len(details)} 题在评测中抛异常（是脚本或系统的错，不是答错）："
            f"{errors[0]['error'][:100]}"
        )

    for label, floor in thresholds:
        if score < floor:
            failures.append(f"{label} —— 实得 {score:.1f}")

    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            previous = json.loads(baseline_path.read_text(encoding="utf-8"))
            drop = previous.get("accuracy", 0.0) * 100 - score
            if drop > args.tolerance:
                failures.append(
                    f"相对基线退步 {drop:.1f} 分（容差 {args.tolerance}）"
                    f"—— 基线 {previous.get('accuracy', 0) * 100:.1f}，现在 {score:.1f}"
                )
            else:
                print(f"\n对基线：{drop:+.1f} 分（容差 ±{args.tolerance}）")
        else:
            print(f"\n[提示] 基线文件不存在，已跳过比较：{baseline_path}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"report": report, "details": details}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"明细已写入：{args.out}")

    print(f"\n{'=' * 66}")
    if failures:
        print("判定：**未达门槛**")
        for item in failures:
            print(f"  ✗ {item}")
        return 1
    print("判定：**通过**" + ("（无绝对门槛，仅记录）" if not thresholds else ""))
    return 0


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def _inspect(questions: list[Question]) -> int:
    """**先看加载器读出了什么，再信任何分数。**

    加载器默默少读 30% 的题，分数会偏低而无从察觉——然后有人会去"优化系统"，
    去修一个根本不存在的问题。
    """
    print(f"\n{'=' * 66}\n加载器读出（前 5 题）\n{'=' * 66}")
    for question in questions[:5]:
        print(f"\n[{question.id}] ({question.category}, 来源 {question.source})")
        print(f"  素材：{len(question.turns)} 段")
        for turn in question.turns[:2]:
            print(f"    - {turn.session_id} @ {turn.ts or '(无时间)'}：{turn.text[:50]}")
        print(f"  问：{question.question}")
        print(f"  答：{question.answer}")
        print(f"  相关关键词：{list(question.gold_keywords)}")
    categories: dict[str, int] = {}
    for question in questions:
        categories[question.category] = categories.get(question.category, 0) + 1
    print(f"\n类别分布：{categories}")
    print(f"合计 {len(questions)} 题")
    return 0


def _print_report(report: dict, details: list[dict], args) -> None:
    print(f"\n{'=' * 66}\n结果\n{'=' * 66}")
    print(f"  题数 {report['total']} · 答题率 {report['answer_rate']:.1%}")
    print(f"  准确率 {report['accuracy'] * 100:.1f} · F1 {report['f1']:.3f} · EM {report['em']:.3f}")
    print(f"  召回 R@{args.top_k} {report['recall@k']:.3f} · MRR {report['mrr']:.3f}")
    print("  ——答题率与准确率的关系就是诊断：答题率高而准确率低 → 修生成；")
    print("    答题率低 → 修检索或提取（与生成无关）。")

    if args.answer_mode == "extract":
        print("\n  ⚠ **extract 模式下的 EM / F1 不可与公开标尺比较。**")
        print("    它把召回内容整段当答案，必然比标准答案长——EM 恒为 0、F1 偏低是")
        print("    **设计使然，不是系统差**。此模式下有解释力的只有三个：")
        print("    答题率（召回走通没有）、R@k（相关内容找到几成）、MRR（排得对不对）。")
        print("    要出可与 LoCoMo ≥ 70 对标的分，必须用 --answer-mode llm。")
        print("    **另外**：检索指标（R@k / MRR）只在**配置了 embedding** 时才代表系统的召回能力；")
        print("    未配置时它们是**纯关键词检索**的基线——不能拿它判断「语义召回好不好」。")

    if report["by_category"]:
        print(f"\n{'=' * 66}\n分类别\n{'=' * 66}")
        for name, bucket in sorted(report["by_category"].items()):
            print(
                f"  {name:<16} n={bucket['total']:<3} 答题率 {bucket['answer_rate']:>6.1%}"
                f" · 准确率 {bucket['accuracy'] * 100:>5.1f} · R@k {bucket['recall@k']:.3f}"
            )

    missed = [d for d in details if not d.get("prediction")]
    if missed:
        print(f"\n未作答 {len(missed)} 题（召回没有走到生成）：")
        for row in missed[:5]:
            print(f"  - {row['id']}：{row['question']}")
    errors = [d for d in details if d.get("error")]
    if errors:
        print(f"\n**异常** {len(errors)} 题（这些是脚本或系统的错，不是答错）：")
        for row in errors[:5]:
            print(f"  - {row['id']}：{row['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="量化门槛执行器（D-13 / G6）")
    parser.add_argument(
        "--dataset",
        default="selfcheck",
        choices=["selfcheck", "locomo", "longmemeval"],
        help="数据集（默认自建集）",
    )
    parser.add_argument("--path", help="数据集文件路径（自建集默认 scripts/bench/cases/selfcheck.jsonl）")
    parser.add_argument(
        "--answer-mode",
        default="extract",
        choices=["extract", "llm"],
        help="extract=零模型抽取式（可进 CI）；llm=真实网关生成",
    )
    parser.add_argument(
        "--ingest",
        default="rule",
        choices=["rule", "llm"],
        help="灌入时用什么提取：rule=规则提取（秒级，把变量隔离到「召回 + 生成」）；"
        "llm=真实 LLM 提取（贵约百倍，但反映完整链路）。"
        "**两者是不同的问题**，混在一起分数就没法归因",
    )
    parser.add_argument(
        "--embedding",
        default="on",
        choices=["on", "off"],
        help="召回要不要用向量：on=配嵌入模型（真实形态）；"
        "off=纯关键词检索。**两者的 R@k 不可直接比较**——"
        "off 反映的是「没有向量时能到哪」，不是「系统的召回能力」",
    )
    parser.add_argument(
        "--home",
        default=None,
        help="把器灵的数据目录固定在这里（默认临时目录、跑完即删）。"
        "**给了它就能复用已灌好的库**——换口径或换作答模型重跑时不必再提取一遍",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="只灌入、不提问。配合 --home 把库准备好，之后反复评测都不用再灌",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 题。**先用它验证链路，再跑全量**——"
        "公开数据集上千题，全量跑一次要几十分钟，链路坏了却要跑完才知道",
    )
    parser.add_argument("--inspect", action="store_true", help="只打印加载器读出的内容，不跑分")
    parser.add_argument("--min-accuracy", type=float, default=None, help="自建集的最小通过分（0-100）")
    parser.add_argument("--baseline", help="基线 JSON（`--out` 产出的），用于判定**不退步**")
    parser.add_argument("--tolerance", type=float, default=2.0, help="允许的退步幅度（分，默认 2）")
    parser.add_argument("--out", help="把明细写成 JSON")
    parser.add_argument("--verbose", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
