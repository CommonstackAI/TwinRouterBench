# Router API 规范（内部）

> 对象：实现 router 的开发者 / SWERouterBench 内部维护者。所有类型定义的单一事实源是 `swerouter/router.py`；本文只做行为说明。

## 1. 核心契约

每跑一个 SWE-bench instance，agent loop 会多次调用 router 的 `select(ctx)`——每当需要发起一次新的 LLM 调用，就回调一次。Router 必须**同步**返回一个 `RouterDecision(model_id=...)`，`model_id` 必须是 `ctx.available_models` 列表中的某一个。

```python
class Router(Protocol):
    def select(self, ctx: RouterContext) -> RouterDecision: ...
```

## 2. `RouterContext` 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `instance_id` | `str` | SWE-bench Verified 的 instance id（例 `django__django-11133`） |
| `step_index` | `int` | 当前是该 instance 的第几步 LLM 调用，从 `0` 开始 |
| `messages` | `list[dict]` | 将要发给 LLM 的完整消息序列（含 system / user / assistant / tool），与后续真正发的请求完全一致 |
| `tools` | `list[dict]` | 本步可用工具的 OpenAI tool schema 列表 |
| `available_models` | `tuple[str, ...]` | 官方模型池（`data/model_pool.json`）的不可变快照 |
| `cache_state` | `CacheStateSnapshot` | 每个 `model_id` 的"**上次调用时间戳 / 前缀 token 数 / 前缀 hash**"只读视图，供 router 做"要不要续 cache"的启发判断 |
| `budget_so_far_usd` | `float` | 本 instance 至今为止 router 已经花掉的美金数（用于 budget-aware router） |
| `run_config` | `RunConfig` | 只读，含 `max_steps`、`budget_usd`、`wallclock_ttl_sec` 等本次 run 的参数 |

`RouterContext` 是**冻结的 dataclass**（`frozen=True`）。Router **不得**修改它。

## 3. `RouterDecision` 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_id` | `str` | **必填**，必须 ∈ `ctx.available_models`，否则 harness 在收到返回值时立即 `raise ValueError`（fail fast） |
| `rationale` | `str \| None` | 可选，router 内部的解释文本，写入 trace 便于事后分析；不参与打分 |

**SWERouterBench 不提供**"回退到另一个模型"的重试机制。若 router 想实现重试，必须自己在 `select` 内多次计算后返回一个最终 `model_id`；一旦返回，harness 就会用那个 `model_id` 发一次请求。

## 4. Fail-fast 行为清单

以下情况 harness **立即 raise**，当前 instance 被标记为 error，不 fallback、不兜底（符合用户规则"严禁错误掩盖"）：

| 情形 | 抛出异常 |
|---|---|
| `RouterDecision.model_id` 不在 `available_models` 内 | `ValueError` |
| `RouterDecision.model_id` 是 `None` / 空字符串 | `ValueError` |
| `router.select` 抛出任何异常 | 原样透传 |
| `router.select` 返回非 `RouterDecision` 对象 | `TypeError` |
| `router.select` 超过 `select_timeout_sec`（默认 30s） | `TimeoutError` |

Instance 抛错时，`InstanceResult.error` 会被填上，当前 instance 计入"失败实例"，参与失败账单的 1× baseline 惩罚。

## 5. 参考实现

### 5.1 `FunctionRouter`（包装任意 callable）

```python
from swerouter.router import FunctionRouter, RouterContext, RouterDecision

router = FunctionRouter(
    label="always_deepseek",
    func=lambda ctx: "deepseek/deepseek-v3.2",      # 返回 str 自动包成 RouterDecision
)
```

允许 `func` 直接返回 `str`（model_id）或 `RouterDecision` 对象。任何其它返回值类型 raise。

### 5.2 内置 baseline（`swerouter.routers`）

- `AlwaysModelRouter(model_id)`：每步恒选同一个 model_id。
- `RoundRobinRouter(model_ids)`：按 `step_index` 轮询。
- `TierFromCRBRouter`：调用 CRB 的 `OpenAICompatRouterClassifier` 出 0–3 档位，再按 `data/tier_to_model.json` 映射到 pool 内 model_id。

## 6. Router 不能看到什么

为了保证 router 只基于"**生产环境真的能拿到**"的输入做决策，`RouterContext` **不**包含：

- 金标 `resolved`（跑完才知道）
- 后续 step 的 messages
- 其它 router 在同一 instance 上的预测
- SWE-bench Verified dataset 里的 `patch` / `test_patch`（ground truth diff）

如果 router 需要"看一眼 query"做分类，它应该从 `ctx.messages` 里抽特征（和 CRB `question_bank_messages_to_classifier_prompt` 的风格一致）。

## 7. 并发与状态

Router 实例可能被 `run_eval` 的多个 instance 并发调用。实现时：

- 若 router 无状态（纯函数）→ 默认安全。
- 若 router 维护状态（例如学习型 router 的内部 KV）→ 实现者负责线程安全；SWERouterBench 不为你加锁。

`CacheStateSnapshot` 是每次 `select` 调用的只读快照，由 harness 在调用前构造，router 不用担心它的线程安全。
