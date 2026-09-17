# 量化门槛自测（D-13 / G6）

> **门槛是里程碑的出口条件，不是终局验收**（DES-REV-001 P2-4）。
> LoCoMo ≥ 70 / LongMemEval ≥ 75 的用途是"确认没跑偏"，**不是刷榜**。

## 三层评测，回答三个不同的问题

| 层 | 命令 | 需要什么 | 回答什么 |
|---|---|---|---|
| **加载器自检** | `--inspect` | 零 | 读进来的东西对不对 |
| **机制回归** | `--answer-mode extract` | **零模型** | 提取 / 入库 / 召回 / 时态有没有坏 |
| **端到端标尺** | `--answer-mode llm` | 真实网关 | 能到多少分（对标公开标尺） |

**为什么要分三层**：一个数字如果同时依赖"检索对不对"和"生成好不好"，
掉下去时你**不知道该修哪一半**。`extract` 模式把生成这一半拿掉，
剩下的全部归因于机制——这正是 G6 要的"自证机制有效"。

## 用法

```bash
# ① 永远先跑这个：确认加载器读出的东西是对的
python scripts/bench/run.py --inspect

# ② 机制回归（零模型、秒级、可进 CI）
python scripts/bench/run.py

# ③ 与基线比，判定"没有退步"（自建集不设绝对门槛）
python scripts/bench/run.py --out baseline.json          # 存一份
python scripts/bench/run.py --baseline baseline.json     # 以后每次比

# ④ 端到端标尺（需要真实网关；这一步的分数才能与 LoCoMo ≥ 70 对标）
REALTEST_API_KEY=... python scripts/bench/run.py --answer-mode llm \
    --dataset locomo --path /path/to/locomo10.json --min-accuracy 70
```

退出码：达门槛 / 无退步 / 无异常 → `0`；否则 `1`（可直接当 CI 门禁）。

## 三个必须知道的口径

### 1. `extract` 模式的 **EM / F1 不可与公开标尺比较**

它把召回内容整段当答案，必然比标准答案长——**EM 恒为 0、F1 偏低是设计使然，
不是系统差**。这个模式下有解释力的只有三个：**答题率 / R@k / MRR**。

拿 `extract` 的 F1 去比 LoCoMo ≥ 70，会得出"系统很差"的错误结论。

### 2. `answered` 与 `total` 是两个分母，别混

- `accuracy = correct / **total**` —— 没答上来的题**算错**；
- `recall` 用 `answered` 当分母 —— 它衡量的是"召回到的东西里找对了几成"。

如果 accuracy 也拿 `answered` 当分母，会得到一个漂亮的假分数：
**召回越差、分母越小、准确率越高**。这类指标的退化方向与实际质量相反，
比没有指标更危险。

### 3. 自建集**刻意不设绝对门槛**

它的用途是"向内自证机制没坏"，判据是**相对基线不退步**。
给自建集定死一条线，会诱导人去调题，而不是去修系统。

## 文件

```
scripts/bench/
├── metrics.py               # 判分器（纯函数、零依赖——可比性的前提）
├── datasets.py              # 统一加载（自建集 / LoCoMo / LongMemEval）
├── run.py                   # 执行器
└── cases/selfcheck.jsonl    # 自建评测集（10 题，7 类机制）
```

## 当前状态（2026-09-17）

| 项 | 状态 |
|---|---|
| 判分器 | ✅ 14 项离线门禁（`tests/test_bench_metrics.py`） |
| 自建集机制回归 | ✅ **R@10 = 1.000、MRR = 0.950、答题率 100%**（`tests/test_bench_selfcheck.py`） |
| 数据集获取 | ✅ `fetch.py`——下载后**立刻用真实加载器自检**（见下） |
| **LoCoMo 实跑** | ✅ **1540 题 / 159 秒**，基线存 `locomo-extract.json` |
| **LongMemEval 实跑** | ✅ **500 题 / 19 秒**，基线存 `longmemeval-extract.json` |
| `--answer-mode llm` | ⛔ **未跑**——需要 `REALTEST_API_KEY` |

### 两个数据集的对照（extract 模式、**零模型、无 embedding**）

| 数据集 | 题数 | 答题率 | R@10 | MRR | NDCG |
|---|---|---|---|---|---|
| LoCoMo | 1540 | 100% | **0.421** | 0.794 | 0.736 |
| LongMemEval（oracle） | 500 | 99.4% | **0.779** | 0.749 | 0.757 |

> **`R@10` 差了近一倍，但这不代表 LongMemEval 上系统更强。**
> LoCoMo 每题要面对 **19 个会话**（大量无关内容），而 oracle 版的 haystack
> **只保留证据会话**（2–3 段，都是相关的）。
> **R@k 的差异主要反映"候选池的难度"，不是系统能力。**（`longmemeval_s/_m` 才是完整难度。）

**两个数据集的短板方向一致**，这不可能是数据集的巧合：

| 数据集 | 最差类别 | R@k |
|---|---|---|
| LoCoMo | `temporal` | 0.266 |
| LoCoMo | `multihop` | 0.432 |
| LongMemEval | `multi-session` | 0.474 |

**跨会话 / 多跳类在两个数据集上都垫底** —— 说明那是**系统的真实弱点**，
不是某个数据集的特性。这是本轮评测最有行动价值的一条结论。

### LoCoMo 基线（extract 模式、**零模型、无 embedding**）

```
1540 题 · 答题率 100.0% · R@10 0.421 · MRR 0.794 · NDCG 0.736 · 耗时 159 秒
```

| 类别 | n | R@k |
|---|---|---|
| `single_hop` | 841 | **0.485** |
| `multihop` | 282 | 0.432 |
| `open_domain` | 96 | 0.346 |
| `temporal` | 321 | **0.266** |

**两个立刻可用的读数**：

- **MRR 0.794 而 R@10 只有 0.421** —— 也就是说"**找到的那条通常排得很前**，
  但**相当一部分相关内容根本没进前 10**"。这是"召回覆盖不足"，不是"排序不好"。
  两种病的修法完全不同（前者要调候选生成 / 扩召回，后者才调融合权重）。
- **`temporal` 最差（0.266）** —— 时间类问题的召回是明确短板。

**这份数字怎么读**（三条，缺一条就会误读）：

1. **它不含 embedding** —— 未配嵌入模型时召回退化为**纯关键词检索**。
   所以 R@k 是"关键词基线"，**不是**系统的语义召回能力。
2. **它的准确率只有 1.9%** —— extract 模式把整段原文当答案，
   而 LoCoMo 的答案是短语（`7 May 2023`）。这是**口径问题，不是能力问题**。
3. **它不与 70 分门槛比较** —— `run.py` 已**拒绝**在 extract 模式下套用公开标尺门槛，
   并打屏说明原因。

> **这份基线能回答**：链路通不通、四类题的相对强弱、改动的效果方向。
> **它不能回答**：与公开方案的高下。

### 怎么把 ⛔ 变成 ✅

```bash
python scripts/bench/fetch.py locomo        # 或手动放 data/locomo10.json
python scripts/bench/fetch.py longmemeval   # LongMemEval
REALTEST_API_KEY=... python scripts/bench/run.py \
    --answer-mode llm --dataset locomo --min-accuracy 70 --out baseline.json
```

配 embedding 后**再跑一次 extract 模式**，就能把"语义召回带来的增量"单独量出来——
这是判断"embedding 到底值不值那份钱与延迟"的直接依据。

**`--inspect` 这一步不能跳**：加载器默默少读，会让分数偏低，而你会去修系统。
