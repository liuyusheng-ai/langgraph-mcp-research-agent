# 真实模型评测结果

由 `python eval/run_real_eval.py` 生成。Agent 模型 `qwen3.5:4b`，裁判模型 `gemma3:4b`（不同模型家族，避免自评），均为 Ollama 本地运行、temperature 0。网页搜索：ddgs 免 Key 元搜索（中文走 Bing），结果按查询缓存在 `search_cache.json`，两组配置对同一查询看到相同结果。

| 指标 | Planner + Researcher + Writer | 去掉 Planner（只用原问题检索） |
| --- | --- | --- |
| 要点召回率（25 道可回答题） | 49.6% | 44.2% |
| 　事实题 / 对比题 | 60.3% / 33.3% | 48.5% / 37.8% |
| 引用忠实度（被引资料支持该句） | 99.0%（101 句） | 97.0%（100 句） |
| 引用覆盖率 | 100.0% | 100.0% |
| 出现不存在的引用编号 | 0 份 | 1 份 |
| 不可回答题正确拒答 | 5/5 | 5/5 |
| 可回答题误拒答 | 0/25 | 0/25 |
| 工具调用成功率 / 有效结果率 | 84.4% / 84.4%（90 次） | 100.0% / 96.7%（30 次） |
| 平均证据数 | 6.7 | 2.9 |
| 延迟 P50 / P95 | 19.2 s / 84.5 s | 3.8 s / 5.9 s |
| 平均 token / 题 | 1396 | 708 |

要点召回率由裁判模型逐条判断报告是否表达了标准要点；被拒答的可回答题按 0 计。引用忠实度对每份报告最多取 8 个带引用的句子，判断所引资料是否支持该句。随机 20% 的裁判结果（85 条）导出到 `spot_check.csv`：`reviewer_claude` 列为更强模型的逐条复核（一致率 82.4%，见 README），`human` 列留给人工填写；运行 `--agreement [--column 列名]` 计算一致率。

## 逐题明细：Planner + Researcher + Writer

| ID | 状态 | 要点召回 | 引用忠实 | 证据数 | 耗时 s |
| --- | --- | --- | --- | --- | --- |
| F01 | completed | 2/5 | 7/7 | 7 | 8.4 |
| F02 | completed | 2/5 | 4/4 | 9 | 5.7 |
| F03 | completed | 3/4 | 4/4 | 6 | 13.8 |
| F04 | completed | 2/4 | 4/5 | 8 | 5.1 |
| F05 | completed | 4/5 | 4/4 | 3 | 84.5 |
| F06 | completed | 3/4 | 4/4 | 5 | 6.0 |
| F07 | completed | 1/5 | 5/5 | 7 | 16.8 |
| F08 | completed | 2/4 | 4/4 | 5 | 44.5 |
| F09 | completed | 4/5 | 5/5 | 9 | 41.7 |
| F10 | completed | 2/4 | 4/4 | 8 | 6.0 |
| F11 | completed | 2/5 | 2/2 | 9 | 16.3 |
| F12 | completed | 4/5 | 5/5 | 8 | 37.7 |
| F13 | completed | 3/5 | 4/4 | 9 | 7.4 |
| F14 | completed | 4/4 | 2/2 | 6 | 47.4 |
| F15 | completed | 3/4 | 3/3 | 8 | 17.1 |
| C01 | completed | 1/5 | 4/4 | 9 | 19.2 |
| C02 | completed | 2/5 | 6/6 | 7 | 5.2 |
| C03 | completed | 1/4 | 5/5 | 5 | 6.1 |
| C04 | completed | 1/5 | 4/4 | 6 | 45.6 |
| C05 | completed | 1/5 | 3/3 | 6 | 45.0 |
| C06 | completed | 1/4 | 3/3 | 9 | 20.3 |
| C07 | completed | 1/4 | 4/4 | 8 | 4.0 |
| C08 | completed | 3/4 | 2/2 | 6 | 45.1 |
| C09 | completed | 2/5 | 4/4 | 8 | 6.5 |
| C10 | completed | 2/4 | 4/4 | 3 | 86.9 |
| U01 | insufficient_evidence | - | - | 3 | 82.8 |
| U02 | insufficient_evidence | - | - | 5 | 42.1 |
| U03 | insufficient_evidence | - | - | 3 | 82.4 |
| U04 | insufficient_evidence | - | - | 8 | 28.0 |
| U05 | insufficient_evidence | - | - | 8 | 3.1 |

## 逐题明细：去掉 Planner（只用原问题检索）

| ID | 状态 | 要点召回 | 引用忠实 | 证据数 | 耗时 s |
| --- | --- | --- | --- | --- | --- |
| F01 | completed | 0/5 | 5/5 | 3 | 4.3 |
| F02 | completed | 2/5 | 3/3 | 3 | 2.9 |
| F03 | completed | 3/4 | 5/5 | 3 | 5.0 |
| F04 | completed | 1/4 | 2/3 | 3 | 2.9 |
| F05 | completed | 4/5 | 4/4 | 3 | 3.4 |
| F06 | completed | 2/4 | 5/5 | 3 | 5.6 |
| F07 | completed | 2/5 | 6/6 | 3 | 4.2 |
| F08 | completed | 3/4 | 2/2 | 3 | 2.3 |
| F09 | completed | 4/5 | 6/6 | 3 | 5.7 |
| F10 | completed | 2/4 | 7/7 | 3 | 5.9 |
| F11 | completed | 0/5 | 2/2 | 3 | 3.3 |
| F12 | completed | 3/5 | 3/4 | 3 | 5.6 |
| F13 | completed | 3/5 | 3/4 | 3 | 6.6 |
| F14 | completed | 3/4 | 4/4 | 3 | 4.7 |
| F15 | completed | 1/4 | 3/3 | 3 | 3.4 |
| C01 | completed | 2/5 | 3/3 | 3 | 3.8 |
| C02 | completed | 3/5 | 4/4 | 3 | 3.2 |
| C03 | completed | 1/4 | 4/4 | 3 | 3.6 |
| C04 | completed | 2/5 | 4/4 | 3 | 4.6 |
| C05 | completed | 1/5 | 3/3 | 3 | 3.7 |
| C06 | completed | 0/4 | 3/3 | 3 | 3.9 |
| C07 | completed | 2/4 | 3/3 | 3 | 3.3 |
| C08 | completed | 1/4 | 6/6 | 3 | 4.1 |
| C09 | completed | 3/5 | 4/4 | 3 | 5.9 |
| C10 | completed | 2/4 | 3/3 | 3 | 4.8 |
| U01 | insufficient_evidence | - | - | 0 | 0.0 |
| U02 | insufficient_evidence | - | - | 3 | 1.5 |
| U03 | insufficient_evidence | - | - | 3 | 1.4 |
| U04 | insufficient_evidence | - | - | 3 | 1.2 |
| U05 | insufficient_evidence | - | - | 3 | 1.5 |
