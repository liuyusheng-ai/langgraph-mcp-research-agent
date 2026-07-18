# LangGraph MCP Research Agent

面向企业与行业调研场景的多智能体研究系统。项目使用 LangGraph 编排 Planner、Researcher、Writer 三类角色，将问题拆解、工具检索、证据整理和报告生成串成可追踪工作流，并对调用预算、重复循环、超时重试和无来源生成进行约束。

## 项目亮点

- Planner、Researcher、Writer 3 类 Agent 分工协作，以共享状态传递计划、证据、调用记录和报告。
- 提供网页搜索、本地 RAG、MCP 3 类工具接口；Tavily 和真实 MCP 服务均为可选配置。
- 使用 SQLite 保存运行状态、工具调用、证据与进度，支持按 `run_id` 恢复已完成任务。
- 设置调用预算、重复循环检测、超时重试、无来源拦截 4 类安全控制。
- 从任务完成、工具调用成功率、引用覆盖率 3 个维度输出 Agentic Eval 指标。
- 自动保留证据来源并生成编号引用，资料不足时停止生成结论性报告。
- 默认离线模式可直接运行；接入真实网页搜索或 MCP 服务时，不改变 Agent 主流程。

## 架构

```text
用户问题
  -> Planner：拆解范围与研究计划
  -> Researcher：
       - Web Search
       - Local RAG
       - MCP Tool
       - 调用预算 / 循环检测 / 超时重试
  -> Writer：基于证据生成带引用报告
  -> Agentic Eval：任务完成 / 工具调用 / 引用覆盖
```

## 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python cli.py "LangGraph、RAG 与 MCP 在企业 Agent 中如何协作"
```

默认使用内置演示资料，不需要 API Key。若需要真实网页搜索：

```bash
set TAVILY_API_KEY=your_key
python cli.py "研究企业 Agent 的安全评测方案"
```

运行结果包括 Markdown 报告、三维评测结果和 `run_id`。恢复已完成任务：

```bash
python cli.py "原问题" --run-id <RUN_ID> --resume
```

## MCP 接入

`mcp_client.py` 使用官方 MCP Python SDK，可连接本地 stdio Server：

```python
from mcp_client import MCPClient, MCPServerConfig
from tools import MCPResearchTool, ResearchToolbox

client = MCPClient(
    MCPServerConfig(command="python", args=["your_mcp_server.py"])
)
toolbox = ResearchToolbox([
    MCPResearchTool(client=client, tool_name="search")
])
```

Agent 会统一记录 MCP 工具的状态、耗时、结果数和异常。生产环境还应为 MCP Server 配置独立权限、参数 Schema 白名单和数据脱敏。

## 测试

```bash
pytest -q
```

测试覆盖带来源报告、调用预算、无来源拦截、SQLite 恢复和三维评测。

## 目录说明

```text
agent.py          LangGraph 三角色编排与 Harness
tools.py          Web、本地 RAG、MCP 工具
mcp_client.py     官方 MCP SDK 客户端
checkpoint.py     SQLite 状态保存与恢复
evaluation.py     三维 Agentic Eval
llm.py            可替换的规划与写作模型
cli.py            命令行演示
test_research_agent.py
```

## LangSmith

LangGraph/LangChain 可通过标准环境变量启用 LangSmith 追踪。请只在自己的环境中配置 `LANGSMITH_API_KEY`，不要把 `.env` 或任何密钥提交到仓库。

## 说明

该项目用于学习与作品展示。离线网页结果使用 `demo://` 来源并会在报告中明确保留；只有配置真实搜索服务后，才会访问互联网。README 不把离线演示结果冒充实时资料。

