# SWERouterBench：为"每一步选模型"付真实账单的 benchmark

> 英文版：[blog_intro.md](blog_intro.md)。

TwinRouterBench 的**静态 track**是离线的：它给 router 一份 970 行的对话前缀题库，router 返回 0–3 档位 id，然后我们把预测和金标档位比一下。这对训练 router 和离线研究都有用，但它回答不了生产审阅会议上那个每次都要问的问题：

> "这个 router 上 SWE-bench Verified 跑一遍，到底能修几个 bug、花掉多少真实美金？"

**动态 track**就是来回答这个问题的，它是静态 track 在 SWE-bench 上的端到端版本。

## 它是什么

SWERouterBench 端到端跑完 SWE-bench Verified 的 500 个实例。每一次 LLM 调用，harness 都把控制权交给 router；router 从**官方锁定的模型池**里挑一个具体 `model_id`；harness 调对应厂商、记录 usage、按 5 分钟 wall-clock prompt cache 模拟、每步写 trace。

跑完一个实例后，我们用 **SWE-bench 官方判据**（`FAIL_TO_PASS` 全过 + `PASS_TO_PASS` 不退化——和 SWE-bench Verified 公开 leaderboard 同一把尺子）判断最终 patch 是否 `resolved`。再按公开价把各步 token 折成美元，并按「失败则加 1× high 基线重跑估计」的规则汇总成**排行榜总账单** `total_leaderboard_bill_usd`（不是单纯的实际 API 花销；后者见 `total_router_cost_usd`）。

没有 combined score，没有任何合成质量指标。**美金越少，排名越前。**

## Router 怎么打分

每个实例：

```
passed_instance_bill = Σ router_actual_cost_i
failed_instance_bill = Σ router_actual_cost_i  +  Σ baseline_high_cost_i
                       ↑                          ↑
                       厂商真的收 router 钱的        用池内最贵模型从头再跑一次
                       数字                       （独立重模拟 cache 以贴近真实）
```

`baseline_high_cost_i` 的算法：用 router 真实 run 的每步 prefix / output token 数，对"全程用最贵模型"这个反事实做**独立的单模型 cache 续写模拟**（5 分钟 TTL），再按那个模型的公开价算。这是"假如你不 route、直接上 HIGH 会花多少"的一阶估计，作为失败情况下的自然账单惩罚。

排名键：`total_leaderboard_bill_usd` 升序。一个每题都选最便宜的 router，绝大多数实例会失败，被 1× baseline 惩罚打回到"≈ 全程 HIGH"的账单量级，因此无法靠"纯省钱"刷榜。

## 它不是什么

- **我们不重判 patch**：`resolved` 由 `swebench.harness.run_evaluation` 裁决，和 SWE-bench Verified 公开榜单 ±3% 对齐。
- **不允许自定义定价**：官方池与价格写死在 `data/model_pool.json` / `data/model_pricing.json`，每次 run 留下 `pricing_fingerprint`；提交 leaderboard 的 run 会由 maintainer 按官方价格**在我们机器上重跑**验证后才上榜。
- **不是训练数据集**：trace 是每次 run 的**产物**、`.gitignore` 掉，归跑的人所有。

## 与静态 track 的关系

| 维度 | 静态 track | 动态 track |
|---|---|---|
| 形态 | 静态 970 行题库 | 动态，500 SWE-bench Verified 实例 |
| Router 输出 | 档位 id 0–3 | 锁定池内具体 `model_id` |
| 通过判据 | `pred_tier >= gold_tier`（软代理） | SWE-bench `resolved`（真跑测试） |
| 定价 | 档位名义价 | 各厂商公开真实价 |
| Cache 模型 | 步距 TTL=3 | Wall-clock TTL=300s |
| 头条指标 | `scores_v2.combined_score`（4 维平均） | `total_leaderboard_bill_usd`（惩罚计入后的总账单，USD） |
| 重依赖 | 无 | `swebench` + `docker` |
| PyPI | base 安装 | `[dynamic]` optional extra |

静态 track 告诉你"每一步**应该**是哪个档位"，动态 track 告诉你"你当时决定错了**实际要付多少钱**"。

## 路线图

**0.1.0 先做最小闭环**：公开 4 个 `always-*` baseline（low/mid/mid_high/high）+ ≥1 个 CRB-classifier 混合 router 的 leaderboard 结果，博客里把表贴出来。之后：

- 开 submission 流程（GitHub issue 模板 + maintainer 按官方价重跑）
- 季度 pricing 刷新（每次 diff 审后 bump `pricing_schema_version`）
- 视社区诉求增加 1h extended-cache 变体

维护 router 的同学如果想上榜，开一个 issue 附上 run artefacts 即可，我们会按官方价重跑。

## 相关链接

- [TwinRouterBench README](../../README.md)：安装与 CLI 快速开始
- [TwinRouterBench paper](https://arxiv.org/abs/2605.18859)