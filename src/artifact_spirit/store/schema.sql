-- 器灵 / Artifact Spirit —— 真相源 DDL
-- 对应 LLD-AL3 §5 M4（该节是本文件的**唯一权威副本**）。
-- 改动本文件必须同步 §5 M4，反之亦然——两边逐字对齐。
--
-- 关于 __EMBEDDING_DIM__：
--   vec0 的维度在建表后不可更改，而维度由配置决定（INV-2 / F7），
--   因此这里用占位符，由 migrate(embedding_dim=...) 替换（默认 2560）。
--
-- 关于 PRAGMA：
--   journal_mode / busy_timeout / foreign_keys 均为**连接级**设置，
--   不属于 schema，故由 sqlite_backend._configure_connection() 在每个连接上施加。

-- ============ 元信息 ============
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
-- 约定键：schema_version / spirit_id / spirit_name
--        embedding_model / embedding_dim / vec_capability / created_at

-- ============ 会话 ============
CREATE TABLE IF NOT EXISTS sessions (
  id         TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  ended_at   TEXT,
  status     TEXT NOT NULL DEFAULT 'active',   -- active | committed
  turn_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status, started_at DESC);

-- ============ 工作记忆（会话内活跃组块）============
CREATE TABLE IF NOT EXISTS working_memory (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  chunk_key    TEXT NOT NULL,
  content      TEXT NOT NULL,
  salience     REAL NOT NULL DEFAULT 0.0,
  act_count    INTEGER NOT NULL DEFAULT 1,
  created_at   TEXT NOT NULL,
  last_touched TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory(session_id, last_touched DESC);

-- ============ 意图槽（前瞻记忆）============
CREATE TABLE IF NOT EXISTS intents (
  id          TEXT PRIMARY KEY,
  session_id  TEXT,
  content     TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'open',    -- open | done | cancelled
  due_at      TEXT,
  created_at  TEXT NOT NULL,
  resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status, due_at);

-- ============ 统一记忆表（真相源，所有层）============
CREATE TABLE IF NOT EXISTS memories (
  id              TEXT PRIMARY KEY,            -- {abbr}_{ulid}，abbr ∈ epi|sem|pro|cor
  layer           TEXT NOT NULL,               -- episodic | semantic | procedural | core
  type            TEXT NOT NULL,               -- fact|preference|event|entity|skill|identity|soul|intent
  subject         TEXT,
  predicate       TEXT,
  object          TEXT,
  content         TEXT NOT NULL,               -- 自然语言陈述 = 事实来源
  abstract        TEXT,                        -- L0 一句话摘要
  scope           TEXT,                        -- JSON {"type":...,"id":...}
  confidence      REAL NOT NULL DEFAULT 0.7,
  salience        REAL NOT NULL DEFAULT 0.0,
  -- 活性（派生字段，可从 audit 重放重建 —— INV-12）
  strength        REAL NOT NULL DEFAULT 1.0,
  access_count    INTEGER NOT NULL DEFAULT 0,
  last_access_at  TEXT,
  -- 时态（Graphiti 式双时态；MVP 只写 valid_from，valid_to 待 M3 的 INVALIDATE）
  valid_from      TEXT,
  valid_to        TEXT,
  -- 来源
  source_session  TEXT,
  source_turn     INTEGER,
  -- 生命周期（D-17 / D-23：两级 + 终态，取消 archived）
  status          TEXT NOT NULL DEFAULT 'active',  -- active | dormant | forgotten
  superseded_by   TEXT,
  -- 审计
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  -- 向量一致性（INV-2）
  embedding_model TEXT,
  -- 内容寻址指纹（D-21 / INV-14）
  content_hash    TEXT,
  FOREIGN KEY (superseded_by) REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_mem_layer_status ON memories(layer, status);
CREATE INDEX IF NOT EXISTS idx_mem_subject      ON memories(subject);
CREATE INDEX IF NOT EXISTS idx_mem_updated      ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_mem_access       ON memories(last_access_at);
CREATE INDEX IF NOT EXISTS idx_mem_valid        ON memories(valid_from, valid_to);
-- D-21：非唯一索引——允许相同内容在不同情境下共存，幂等由导入逻辑保证
CREATE INDEX IF NOT EXISTS idx_mem_chash        ON memories(content_hash);

-- ============ 全文检索 ============
-- 实现决策（偏离 LLD-AL3 的"外部内容表模式"）：
--   外部内容表要求索引内容与 memories 逐字一致，而器灵需要对中文做逐字归一化
--   （见 store/text.py 文件头）。因此改为**普通 FTS5 表**，由触发器写入归一化文本。
--   它仍是**派生索引**，可用 rebuild_fts() 从 memories 重建（INV-1）。
--   spirit_norm() 是连接级注册的 Python 函数（sqlite_backend._register_functions）。
CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(
  content, abstract, subject, object,
  tokenize='unicode61 remove_diacritics 2'
);

-- ============ 向量索引（维度由配置决定，见文件头说明）============
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
  mem_id    TEXT PRIMARY KEY,
  embedding float[__EMBEDDING_DIM__]
);

-- ============ 实体节点（语义记忆图）============
CREATE TABLE IF NOT EXISTS entities (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  type       TEXT,                             -- person|project|org|concept|place
  aliases    TEXT,                             -- JSON array
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ent_name_type ON entities(name, type);

-- ============ 关联边（共激活 / 派生 / 提及 / 取代）============
CREATE TABLE IF NOT EXISTS relations (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  src_kind   TEXT NOT NULL,                    -- memory | entity
  src_id     TEXT NOT NULL,
  dst_kind   TEXT NOT NULL,
  dst_id     TEXT NOT NULL,
  rel_type   TEXT NOT NULL,                    -- co_activation|hebbian|derived_from|mentions|supersedes
  weight     REAL NOT NULL DEFAULT 0.0,
  co_count   INTEGER NOT NULL DEFAULT 1,
  valid_from TEXT,
  valid_to   TEXT,
  last_co_at TEXT,
  UNIQUE (src_kind, src_id, dst_kind, dst_id, rel_type)
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relations(src_kind, src_id, weight DESC);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON relations(dst_kind, dst_id, weight DESC);

-- ============ L1 概览缓存（分级加载 · M9）============
-- 注意：这是**缓存**不是真相源（INV-1）——可从底层记忆重算，清空无害
CREATE TABLE IF NOT EXISTS overviews (
  id           TEXT PRIMARY KEY,
  scope_kind   TEXT NOT NULL,                  -- entity | topic | layer
  scope_id     TEXT NOT NULL,
  level        TEXT NOT NULL DEFAULT 'L1',
  content      TEXT NOT NULL,
  token_count  INTEGER,
  generated_at TEXT NOT NULL,
  stale        INTEGER NOT NULL DEFAULT 0,     -- 底层变更后置 1，下次读取重算
  model        TEXT,
  UNIQUE (scope_kind, scope_id, level)
);
CREATE INDEX IF NOT EXISTS idx_ov_stale ON overviews(stale, generated_at);

-- ============ 审计（append-only —— INV-8 / D-22）============
CREATE TABLE IF NOT EXISTS audit (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  ts          TEXT NOT NULL,
  op          TEXT NOT NULL,                   -- add|update|set_status|touch|link|reinforce|forget|restore|consolidate|summarize|evolve|reindex|reconcile
                                               -- 权威列举在 base.AuditEvent；本列**无枚举校验**（评审 P1-20）
  actor       TEXT NOT NULL,                   -- extractor|dedup|consolidator|decay|optimizer|user|cli|system
  target_kind TEXT,                            -- memory|relation|entity|session|intent
  target_id   TEXT,
  before      TEXT,                            -- JSON 快照（D-22：删除时必填，用于恢复）
  after       TEXT,                            -- JSON 快照
  reason      TEXT,
  session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts     ON audit(ts);
CREATE INDEX IF NOT EXISTS idx_audit_target ON audit(target_kind, target_id);

-- ============ 删除快照（D-22 安全网）============
-- 为什么单独一张表，而不是塞进 audit.before：
--   合规删除要求"数据真的不存在"，须清除内容快照；而 audit 是 append-only（INV-8），
--   不可 UPDATE/DELETE。把**内容**（可清除）与**账本**（永不删）分开，两个约束才不打架：
--     audit            = 谁 / 何时 / 因何 / 对哪条，永不删
--     delete_snapshots = 恢复用的内容快照，可被合规清除
CREATE TABLE IF NOT EXISTS delete_snapshots (
  audit_id   INTEGER PRIMARY KEY,           -- 对应 audit.id
  mem_id     TEXT NOT NULL,
  payload    TEXT NOT NULL,                 -- JSON：MemoryRecord + 关联边
  created_at TEXT NOT NULL
);

-- ============ 触发器：FTS5 同步（普通 FTS5 表模式，见上方 mem_fts 说明）============
CREATE TRIGGER IF NOT EXISTS trg_mem_ai AFTER INSERT ON memories BEGIN
  INSERT INTO mem_fts(rowid, content, abstract, subject, object)
  VALUES (new.rowid, spirit_norm(new.content), spirit_norm(new.abstract),
          spirit_norm(new.subject), spirit_norm(new.object));
END;

CREATE TRIGGER IF NOT EXISTS trg_mem_ad AFTER DELETE ON memories BEGIN
  DELETE FROM mem_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS trg_mem_au AFTER UPDATE ON memories BEGIN
  DELETE FROM mem_fts WHERE rowid = old.rowid;
  INSERT INTO mem_fts(rowid, content, abstract, subject, object)
  VALUES (new.rowid, spirit_norm(new.content), spirit_norm(new.abstract),
          spirit_norm(new.subject), spirit_norm(new.object));
END;

-- ============ 触发器：审计 append-only（INV-8）============
CREATE TRIGGER IF NOT EXISTS trg_audit_no_update BEFORE UPDATE ON audit BEGIN
  SELECT RAISE(ABORT, 'audit is append-only (INV-8)');
END;

CREATE TRIGGER IF NOT EXISTS trg_audit_no_delete BEFORE DELETE ON audit BEGIN
  SELECT RAISE(ABORT, 'audit is append-only (INV-8)');
END;
