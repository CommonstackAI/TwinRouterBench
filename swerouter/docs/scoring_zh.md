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

## 2. 账单公式（v2 — 固定机会成本惩罚）

### 2.1 单实例

```
FAILURE_PENALTY_USD = 0.60   # 假定完美求解器的每 case 定价

if instance.resolved == True:
    instance_bill = Σ_i router_actual_cost_i          # 只付真实 router 成本

if instance.resolved == False:
    instance_bill = Σ_i router_actual_cost_i + FAILURE_PENALTY_USD
                    ↑                           ↑
                    该条 trace 全长真实成本       固定机会成本 / 补救惩罚
```

**为什么是固定 \$0.60 而非动态模拟？**
- 和步数完全脱钩：不会出现「agent 磨很久 → 惩罚爆炸」。
- 和定价表弱耦合：不用维护等效步数缓存重放逻辑。
- \$0.60 建模为假定完美求解器（100% resolve）的每 case 定价，统一适用于所有 policy（含无路由 baseline）。

### 2.2 汇总

```
total_leaderboard_bill_usd = Σ_instance instance_bill
                           = Σ router_actual + 0.60 × #unresolved

total_router_cost_usd      = Σ_instance Σ_i router_actual_cost_i   # 辅助列：纯路由 API 真实支出
total_penalty_cost_usd     = 0.60 × #unresolved                    # 辅助列
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

## 4. 边界情况

| 情形 | 处理 |
|---|---|
| instance 直接 error（router 抛异常 / harness 崩） | 视为 `resolved=False`，收 \$0.60 固定惩罚；router_cost 只计到崩溃前那一步 |
| instance 超 `max_steps` 但没调 finish tool | 视为 `resolved=False` |
| instance 超预算 `budget_usd` 主动终止 | 同上 |
| router 全 instance 返回同一 HIGH model_id | resolved → 只付真实支出；失败 → 真实支出 + \$0.60 |
| 空 trace（0 步 instance） | 视为 `resolved=False`；`instance_bill = 0 + 0.60 = $0.60`。Harness 应 raise（空 trace 是 bug） |

## 6. 跨版本对比

每次 `run_eval` 写入 `eval_summary.json` 的 `pricing_fingerprint` 格式：

```
pricing_fingerprint = f"{pricing_schema_version}.{pool_schema_version}.{ttl_schema_version}"
```

不同 `pricing_fingerprint` 的 run **不可直接合到同一 leaderboard**；`swerouter.leaderboard.render` 遇到混版本必须显式分组或拒绝。这样避免"偷偷改定价让某条 baseline 变便宜"这种隐性作弊。
