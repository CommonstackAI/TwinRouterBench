# Leaderboard 打分规则（内部）

> 单一事实源：`swerouter/leaderboard/score.py`。本文解释公式与设计动机。

## 1. 单一指标

```
leaderboard 主排序键 = total_leaderboard_bill_usd  （越低越好）
```

字段名 **`total_leaderboard_bill_usd`**（JSON / `score.json`）表示「用于排名的**总账单**」：在 **resolved** 时等于真实路由 API 支出之和；在失败实例上会**加上** 1× high-baseline 重跑估计。它**不是**「实际花销」的同义词——纯路由真实支出请看 **`total_router_cost_usd`**。旧版曾误用 `total_actual_bill_usd`，易与「实际 API 账单」混淆；`swerouter/leaderboard/render.py` 仍可读该旧键以兼容历史 `score.json`。

**不做 combined_score**（不像 CRB v2 那种 4 项平均）。理由：SWE-bench 的 `resolved` 已经是硬金标，router 不能靠"错得便宜"作弊——失败 instance 直接触发 1× baseline 惩罚，所以"一个 router 如果便宜但失败多"会被自动拖到榜尾。单一美元指标工业可读性最高。

辅助列（**不参与排序**，但必须展示，便于人看）：

- `resolved_count` / `resolved_rate`
- `total_router_cost_usd`（router 真实 API 账单）
- `total_penalty_cost_usd`（失败 instance 额外扣的 1× baseline 总和）
- `avg_steps`、`avg_cost_per_resolved`、`pricing_fingerprint`

## 2. 账单公式

### 2.1 单实例

令一条 instance 有 `n` 步 LLM 调用。记第 `i` 步（`i = 0, 1, ..., n-1`）的 router 真实账单为 `router_actual_cost_i`，baseline（用 HIGH 模型以该步实际 token 重跑）账单为 `baseline_high_cost_i`：

```
if instance.resolved == True:
    instance_bill = Σ_i router_actual_cost_i

if instance.resolved == False:
    instance_bill = Σ_i router_actual_cost_i  +  Σ_i baseline_high_cost_i
                    ↑                             ↑
                    实际 API 账单                 1 × 用 HIGH 从头重跑（失败惩罚）
```

**惩罚倍数恰好为 1**。CRB v2 用的是 2×（多扣一次等价惩罚），但 SWERouterBench 的 `resolved` 是硬金标、不会产生 CRB 里那种"错得便宜"倒挂，所以只取自然账单的 1×。详见 CRB `doc/router_scoring_v2_design_and_ablation_zh.md` §5.2 与本项目 `docs/design_zh.md` §1 对比表。

### 2.2 汇总

```
total_leaderboard_bill_usd  = Σ_instance instance_bill
total_router_cost_usd  = Σ_instance Σ_i router_actual_cost_i        # 辅助列：真实路由 API 支出
total_penalty_cost_usd = Σ_instance (1-resolved) × Σ_i baseline_high_cost_i  # 辅助列
```

## 3. `router_actual_cost_i` 怎么算

每步由 `swerouter.pricing.step_real_cost_usd(usage, pricing)` 计算，`usage` 是 `swerouter.usage.normalize_usage` 归一后的 4 桶：

```
router_actual_cost_i = (usage.input_tokens     × pricing.input_per_m
                      + usage.cache_read_tokens × pricing.cache_read_per_m
                      + usage.cache_write_tokens × pricing.cache_write_per_m
                      + usage.output_tokens    × pricing.output_per_m) / 1_000_000
```

定价按 `RouterDecision.model_id` 从 `data/model_pricing.json` 查。未知 model_id → raise。

## 4. `baseline_high_cost_i` 的 Cache 独立重模拟

Baseline 是"**全程用 HIGH 模型**"的反事实账单。因为 HIGH 全程单模型，缓存会完美续写（不会因 router 切模型而 miss），所以必须**独立重新模拟**一遍缓存状态，不能复用 router 真实 run 里的 cache hit/miss。

令 `prefix_tokens_i` 是第 `i` 步 input 的总 token 数（= `usage.input + usage.cache_read + usage.cache_write`，从 trace 里读），`output_tokens_i` 同理，`wallclock_ts_i` 是该步 `started_at`：

```
for i in 0..n-1:
    cold_start = (i == 0) or (wallclock_ts_i - wallclock_ts_{i-1} > wallclock_ttl_sec)
    if cold_start:
        cache_read_tok_i  = 0
        cache_write_tok_i = prefix_tokens_i
    else:
        cache_read_tok_i  = prefix_tokens_{i-1}
        cache_write_tok_i = max(0, prefix_tokens_i - prefix_tokens_{i-1})

    baseline_high_cost_i =
        ( cache_read_tok_i  × HIGH.cache_read_per_m
        + cache_write_tok_i × HIGH.cache_write_per_m
        + output_tokens_i   × HIGH.output_per_m
        ) / 1_000_000
```

其中 `HIGH` 从 `data/model_pricing.json` 查 `is_high_baseline=true` 那一项。若 pool 内无 / 多于一个被标为 high baseline → raise。

### 4.1 `prefix_tokens_i - prefix_tokens_{i-1}` 可能为负？

理论上每一步 `messages` 只追加（system / tool result / assistant），不删，所以 `prefix_tokens_i >= prefix_tokens_{i-1}`。但不同模型 tokenizer 不同，token 数可能反而略降；此时取 `max(0, ...)`，**不 raise**（因为这是 baseline 估价阶段的合法近似，不是业务错误）。这是 `docs/pricing_and_cache_zh.md` §4 提到的一阶近似：baseline 直接复用 router run 时各模型原生 tokenizer 数出的 token，不为 HIGH 重切。不同 tokenizer 之间的 ±10-20% 差异对所有 router 同等放大/缩小，不影响排名次序。

## 5. 边界情况

| 情形 | 处理 |
|---|---|
| instance 直接 error（router 抛异常 / harness 崩） | 视为 `resolved=False`，按失败公式；router_cost 只计到崩溃前那一步；baseline 按已有 trace 步数重模拟 |
| instance 超 `max_steps` 但没调 finish tool | 视为 `resolved=False`（patch 可能为空，跑 harness 仍出 resolved=False） |
| instance 超预算 `budget_usd` 主动终止 | 同上，`resolved=False` |
| router 全 instance 返回同一 HIGH model_id | `router_actual_cost ≈ baseline`（失败时仍扣 1×baseline，账单 ≈ 2× baseline；这是预期行为） |
| 空 trace（0 步 instance） | 视为 `resolved=False`；`instance_bill = 0` router + 0 baseline = $0；会被挡在 `docs/design_zh.md` §4 描述的 resolved=False 账单分支但金额为 0。**Harness 应 raise**，因为 SWE-bench 每个 instance 至少要跑出 1 步；空 trace 是 bug |

## 6. 跨版本对比

每次 `run_eval` 写入 `eval_summary.json` 的 `pricing_fingerprint` 格式：

```
pricing_fingerprint = f"{pricing_schema_version}.{pool_schema_version}.{ttl_schema_version}"
```

不同 `pricing_fingerprint` 的 run **不可直接合到同一 leaderboard**；`swerouter.leaderboard.render` 遇到混版本必须显式分组或拒绝。这样避免"偷偷改定价让某条 baseline 变便宜"这种隐性作弊。
