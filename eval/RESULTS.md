# 评测结果

由 `python eval/run_eval.py` 生成。离线运行：网页搜索使用演示资料，MCP 为仓库自带的 stdio 服务，不调用大模型。

| 指标 | 结果 |
| --- | --- |
| 研究任务通过 | 15/15 |
| 可回答任务完成率 | 100% |
| 引用覆盖率 | 100% |
| 工具调用成功率 | 100% |
| 有效工具结果率 | 75% |
| 无资料问题正确停止 | 3/3 |
| Harness 场景通过 | 8/8 |
| 单任务延迟 P50 / P95 | 347.79 ms / 434.71 ms |

## 研究任务

| ID | 问题 | 状态 | 证据数 | 引用覆盖 | 耗时 ms | 通过 |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | LangGraph RAG MCP Agent 工程实践 | completed | 7 | 100% | 434.71 | ✅ |
| R02 | LangGraph 如何组织多智能体工作流 | completed | 3 | 100% | 350.43 | ✅ |
| R03 | RAG 如何保存来源并避免无依据回答 | completed | 3 | 100% | 351.8 | ✅ |
| R04 | MCP 工具调用前需要做哪些校验 | completed | 5 | 100% | 352.07 | ✅ |
| R05 | MCP 服务的权限与脱敏怎么设计 | completed | 2 | 100% | 347.51 | ✅ |
| R06 | 多智能体 Planner Researcher Writer 如何分工 | completed | 1 | 100% | 347.79 | ✅ |
| R07 | Agent 评测关注哪些指标 | completed | 3 | 100% | 346.69 | ✅ |
| R08 | LangGraph 任务中断后如何断点恢复 | completed | 4 | 100% | 348.58 | ✅ |
| R09 | 企业 Agent 项目的安全控制与回归测试 | completed | 3 | 100% | 347.84 | ✅ |
| R10 | RAG 与 MCP 在企业 Agent 中如何协作 | completed | 7 | 100% | 348.89 | ✅ |
| R11 | 共享状态如何在 Agent 节点之间传递证据 | completed | 6 | 100% | 173.09 | ✅ |
| R12 | 工具调用的参数和超时如何验证 | completed | 6 | 100% | 176.46 | ✅ |
| R13 | 今年大豆期货价格走势 | insufficient_evidence | 0 | 0% | 174.11 | ✅ |
| R14 | 北京明天的天气 | insufficient_evidence | 0 | 0% | 174.89 | ✅ |
| R15 | 足球比赛比分预测 | insufficient_evidence | 0 | 0% | 174.64 | ✅ |

## Harness 场景

| ID | 场景 | 预期 | 实际 | 通过 |
| --- | --- | --- | --- | --- |
| H01 | budget | 预算为 2 时只执行 2 次工具调用，并记录 budget_exhausted | calls=2 | ✅ |
| H02 | loop | 规划出重复子问题时，第二次同工具同参数调用被 loop_blocked，不消耗预算 | ['success', 'loop_blocked'] | ✅ |
| H03 | timeout | 工具超时后重试一次，最终状态 timeout，不生成报告 | timeout attempts=2 | ✅ |
| H04 | transient_retry | 临时错误重试后成功，attempts=2 | attempts=2 | ✅ |
| H05 | permanent_error | 永久错误不重试，attempts=1 | attempts=1 | ✅ |
| H06 | mcp_not_configured | 未配置 MCP 记为 not_configured，不计入调用成功率 | {'local_rag': 'success', 'mcp': 'not_configured'} | ✅ |
| H07 | no_evidence | 没有证据时停止生成，输出资料不足 | insufficient_evidence | ✅ |
| H08 | resume | 中断后按 run_id 恢复，已完成的调用不重复执行 | a=0 b=1 | ✅ |
