# 真实定价与 Prompt Cache 模型（内部）

> 单一事实源：`swerouter/pricing.py`、`swerouter/cache.py`、`swerouter/usage.py` 以及 `data/` 下三份 JSON。本文件只解释**为什么这样设计**和**字段怎么填**。

## 1. 三份配置文件

所有业务值外置，代码内**不许**硬编码（符合用户规则 1.1）。

### 1.1 `data/model_pool.json`

官方允许的 model_id 白名单 + 每家 provider 的 API 风格。**锁死**，不允许 router 自定义池——防作弊。

```json
{
  "schema_version": 1,
  "pool": [
    {
      "model_id": "anthropic/claude-opus-4.6",
      "provider": "anthropic",
      "is_high_baseline": true
    },
    {
      "model_id": "google/gemini-3-flash-preview",
      "provider": "openai_compat"
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `model_id` | API 侧发给 `base_url/v1/chat/completions` 的 model 名，必须和 `model_pricing.json` 里的 key 一一对应 |
| `provider` | `anthropic` / `openai` / `deepseek` / `gemini` / `openai_compat`（走 OpenRouter 等聚合） |
| `is_high_baseline` | 布尔，必须且**仅有一个** model 被标 `true`，标记池内「最贵 / 全 Opus」参考模型（报告、基线对照）；**排行榜失败惩罚**为 `leaderboard_penalty.json` 中的固定美元项，见 `docs/scoring_zh.md` |

### 1.2 `data/model_pricing.json`

每个 model_id 的 4 桶真实价，锁死版本号。

```json
{
  "schema_version": 1,
  "fetched_at": "2026-04-17",
  "notes": "Prices in USD per 1M tokens. Cache write fallback to base input price where provider does not publish separate cache_write.",
  "pricing": {
    "anthropic/claude-opus-4.6": {
      "input_per_m": 5.00,
      "output_per_m": 25.00,
      "cache_read_per_m": 0.50,
      "cache_write_per_m": 6.25,
      "source_url": "https://www.anthropic.com/pricing"
    }
  }
}
```

| 字段 | 说明 |
|---|---|
| `schema_version` | 整数，每次 pricing 数值变更必须 bump；写入每次 run 的 `eval_summary.json` 便于事后复核 |
| `fetched_at` | ISO 日期；`scripts/refresh_pricing.py` 自动填 |
| `pricing[model_id].input_per_m` | 裸 input token 单价（$/M） |
| `pricing[model_id].output_per_m` | 输出 token 单价（$/M） |
| `pricing[model_id].cache_read_per_m` | cache hit 时复用部分的单价（$/M） |
| `pricing[model_id].cache_write_per_m` | 首次写入 prompt cache 的单价（$/M）；若厂商未公开，**保守**取 `input_per_m` |
| `pricing[model_id].source_url` | 可验证的价目表 URL，支持 OpenRouter `models/{id}` 或各厂商定价页 |

**未知 model_id**：`swerouter.pricing.load_pricing_table` 与 `step_real_cost_usd` 都会 raise；**不做静默 fallback**。

### 1.3 `data/ttl_policy.json`

官方唯一的缓存 TTL 策略。

```json
{
  "schema_version": 1,
  "policy_name": "WALLCLOCK_5MIN",
  "wallclock_ttl_sec": 300
}
```

选 300 秒的原因：对齐 Anthropic ephemeral cache、OpenAI automatic cache 的默认失效时间。Router 不应假设更长 TTL；若厂商允许 `extended` 1h cache，需要显式在 `llm_client` 配置并在 `ttl_policy.json` bump schema 重跑 baseline。

## 2. 4 桶 Usage 归一

不同厂商的 `usage` 字段差异很大。SWERouterBench 在 `swerouter.usage.normalize_usage(provider, raw_usage)` 里统一归到 4 桶：

```python
@dataclass(frozen=True)
class UsageBuckets:
    input_tokens: int         # 未命中 cache 且不是本次新写 cache 的裸 input
    cache_read_tokens: int    # 命中 cache 的 input（读价）
    cache_write_tokens: int   # 本次新写入 cache 的 input
    output_tokens: int        # completion / assistant 新生成
```

映射规则：

| 厂商 | raw 字段 → 4 桶 |
|---|---|
| **OpenAI** | `input = prompt_tokens - cached_tokens - cache_write_tokens`（均在 `prompt_tokens_details` 内，后者缺省为 0）；`cache_read = cached_tokens`；`cache_write = cache_write_tokens`（OpenRouter 转发 Anthropic ephemeral 时会填）；`output = completion_tokens` |
| **Anthropic** | `input = input_tokens`；`cache_read = cache_read_input_tokens`；`cache_write = cache_creation_input_tokens`；`output = output_tokens` |
| **DeepSeek** | `input = prompt_cache_miss_tokens`；`cache_read = prompt_cache_hit_tokens`；`cache_write = 0`；`output = completion_tokens` |
| **Gemini** | `input = prompt_token_count - cached_content_token_count`；`cache_read = cached_content_token_count`；`cache_write = 0`；`output = candidates_token_count` |
| **openai_compat（OpenRouter 聚合）** | 与 **OpenAI** 同一归一函数；须消费 `prompt_tokens_details.cache_write_tokens`，否则 Claude 类缓存写会被误算进 `input` 桶，与 OpenRouter `usage.cost` 不一致 |

**未知 provider** → raise。**任何桶值为负或非整数** → raise（用户规则 1.3 fail fast）。

### 2.1 Trace 里的 ``litellm_estimate``（审计用，不参与计费）

每步 JSONL 除 ``usage`` / ``raw_usage`` / ``step_cost_usd`` 外，可带 ``litellm_estimate``：用 **LiteLLM ``token_counter``**（``openrouter/<model_id>``）对**本步请求前**的 ``messages`` 与 assistant 回复分别估 prompt / completion token 数，并给出 **naive_cost_usd**（整段 prompt 估数按 ``input_per_m``、completion 估数按 ``output_per_m``，**不**拆分 cache_read / cache_write）。用于和网关 ``usage``、``raw_usage.cost`` 对照；**榜单计费仍以 ``usage`` 四桶 + ``model_pricing.json`` 为准**。

## 3. 实际 run 的 Cache 判定（`swerouter.cache`）

Agent loop 在真调 LLM 前，先用 `PromptCacheModel.lookup(model_id, messages, now_ts)` 判当前请求 **相对 harness 视角**是 hit 还是 miss。这个判定**只用于 Router 的 `cache_state` 字段**——真实 API 侧是否命中由厂商决定并在返回 `usage` 里体现。

判定规则：

1. 同一 instance 内维护 `dict[model_id -> (last_call_ts, prefix_hash, prefix_token_count)]`。
2. 语义前缀匹配：当前 `messages` 的前 N 条必须和上次调用的 `messages` 在 `role`、`content`、`tool_calls`、`tool_call_id`、`name` 上完全一致（比较时忽略 `cache_control` 块）。复用 CRB `main/eval/section11.py` 里已有的 `_semantic_prefix_len` 等工具。
3. Wall-clock TTL：`now_ts - last_call_ts <= wallclock_ttl_sec`。
4. 任何一条不满足 → miss，整段 prompt 视为 cache_write。

## 4. 与排行榜账单的关系

**`total_leaderboard_bill_usd` 不再**对失败实例做「按真实步数用 HIGH 模型 + 独立 cache 重模拟」的反事实重算；失败条仅加固定 `C_usd`（见 `docs/scoring_zh.md`）。本文档的 cache / 4 桶价仍用于 **trace 上真实 router 调用**的计费与诊断。

## 5. Pricing schema 版本治理

- `scripts/refresh_pricing.py` 拉最新价与 `data/model_pricing.json` 对比 → 若有 diff，**必须人工 review** 决定是否 bump `schema_version`。
- 每次 `run_eval` 把 `schema_version + fetched_at + pool hash` 写进 `eval_summary.json` 的 `pricing_fingerprint` 字段。
- 跨 schema 版本的 run 之间**不能直接对比总账单**；leaderboard 渲染器在同一张表里遇到不同 `pricing_fingerprint` 应该拒绝合并或显式分组。
