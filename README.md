# AIHot MCP Python 服务

这是一个基于 FastAPI 的轻量网关，通过 MCP Streamable HTTP 连接 AIHot，并把资讯、搜索、热点、事件时间线和每日简报能力转换成普通 HTTP API。

## 环境要求

- Python 3.10+

## 启动

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env  # Windows 可手动复制
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后打开 `http://127.0.0.1:8000/docs` 可直接调试。

## 使用流程

1. 查看 AIHot 提供的工具及参数结构：

```bash
curl http://127.0.0.1:8000/tools
```

2. 根据上一步返回的工具名称和 `inputSchema` 调用：

```bash
curl -X POST http://127.0.0.1:8000/call/工具名称 \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"参数名": "参数值"}}'
```

也可以使用统一入口：

```bash
curl -X POST http://127.0.0.1:8000/fetch \
  -H "Content-Type: application/json" \
  -d '{"tool": "工具名称", "arguments": {"参数名": "参数值"}}'
```

## 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/health` | 服务存活检查 |
| GET | `/tools` | 获取 MCP 工具列表和输入结构 |
| POST | `/call/{tool_name}` | 调用指定 MCP 工具 |
| POST | `/fetch` | 通过请求体指定工具并获取信息 |
| GET | `/news/latest` | 获取 24 小时或 7 天最新 AI 资讯 |
| POST | `/news/search` | 搜索最近 7 天的 AI 资讯 |
| GET | `/topics/hot` | 获取当前 AI 热点榜 |
| GET | `/stories/{public_id}` | 获取热点事件时间线 |
| GET | `/daily` | 获取最新或指定日期的每日简报 |

常用示例：

```bash
# 最近 24 小时精选资讯
curl "http://127.0.0.1:8000/news/latest?window=24h&mode=selected&limit=10"

# 搜索资讯
curl -X POST http://127.0.0.1:8000/news/search \
  -H "Content-Type: application/json" \
  -d '{"q":"OpenAI","window":"7d","limit":10}'

# 当前热点榜
curl "http://127.0.0.1:8000/topics/hot?limit=10"

# 最新每日简报；指定日期时使用 /daily?report_date=2026-09-08
curl http://127.0.0.1:8000/daily
```

AIHOT MCP 为匿名只读接口，默认直接使用 `https://aihot.news/api/mcp`，无需 API Key。兼容旧部署时仍可选配 `AIHOT_ACTOR`，但新接入无需设置。

## Docker

```bash
docker build -t aihot-mcp-service .
docker run --rm -p 8000:8000 --env-file .env aihot-mcp-service
```

## 每周使用 Qwen 发现新增模型并补充 Artificial Analysis 指标

模型追踪器不再维护发布关键词、正则表达式或固定模型家族列表。执行流程为：

1. `qwen3.8-flash` 通过原生工具调用请求 AIHOT MCP 的 `aihot_get_latest`；
2. AIHOT 查询固定覆盖最近 7 天的全部模型分类资讯；
3. Qwen 根据标题和摘要判断是否属于已正式发布、上线 API 或开放权重的新模型，并输出结构化 JSON；
4. 程序核验每个结果引用的 AIHOT 条目 ID，只接受真实来源，再进行确定性去重和落盘；
5. 启用 Artificial Analysis 时，程序使用原有本地规则完成模型名称到 AA slug 的匹配，再抓取并格式化指标。

MCP 查询参数如下：

```json
{"window":"7d","mode":"all","category":"ai-models","limit":30}
```

因此，新出现的模型系列不需要修改代码或补充正则规则。传闻、预告、评测对比、价格变化、旧模型部署以及只发布产品而未发布模型的资讯会由 Qwen 进行语义排除。

先立即运行一次进行测试：

```bash
python -m app.model_tracker --once
```

结果会写入：

- `data/seen_models.json`：已经发现过的模型，用于跨周期去重；
- `data/reports/weekly_models_*.json`：机器可读结果；
- `data/reports/weekly_models_*.md`：便于查看的中文报告。
- `存量模型信息记录状态.json`：新增模型及其最新 AA 三项指标；
- `存量模型信息记录状态.intelligence-index.txt`：上次看到的 Intelligence Index 说明文本。

发现新增模型后，脚本会访问 [Artificial Analysis Models](https://artificialanalysis.ai/models)，为每个模型补充：

- `intelligence_index`：Artificial Analysis Intelligence Index，四舍五入为整数；
- `output_tokens_per_task`：每个 Intelligence Index 任务的输出 token 数除以 1000 后四舍五入为整数，并保存为 `xx K`；
- `time_per_task_minutes`：每个 Intelligence Index 任务的耗时，仅保存分钟并四舍五入到小数点后一位；
- 实际匹配到的 AA 型号、slug、参数量、推理深度和匹配方式。

AA 型号选择保留原有本地规则匹配：依次比较 AA 的型号 slug、完整名称、去除思考深度限定后的名称以及所属 release 家族名称，并结合包含关系和名称相似度选择候选。输入名称只对应一个模型系列时，优先选择参数量最大的型号，参数量相同时选择思考深度最高的变体；无法可靠对应时标记为“未匹配”。该过程不调用 Qwen，因此不会因 AA 映射额外触发模型 API 请求。AA 页面暂时不可用时，本轮模型发现和去重仍会正常完成，报告中的 AA 指标显示为“未获取”。页面显示 `Updated` 且 Intelligence Index 说明文本发生变化时，状态表中的全部模型会先标为“待更新”，再逐项刷新；抓取或匹配失败的记录会保持“待更新”，供下次任务重试。可用 `--status-table` 指定其他状态表路径。

启动常驻任务：

```bash
python -m app.model_tracker --daemon
```

默认每周一北京时间 09:00 执行。可以在 `.env` 中修改：

```dotenv
TRACKER_TIMEZONE=Asia/Shanghai
TRACKER_DAY_OF_WEEK=mon
TRACKER_HOUR=9
TRACKER_MINUTE=0
QWEN_API_KEY=replace-with-your-dashscope-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.8-flash
QWEN_TIMEOUT_SECONDS=120
QWEN_TEMPERATURE=0
QWEN_ENABLE_THINKING=false
QWEN_REQUEST_ATTEMPTS=3
ARTIFICIAL_ANALYSIS_ENABLED=true
ARTIFICIAL_ANALYSIS_BASE_URL=https://artificialanalysis.ai
ARTIFICIAL_ANALYSIS_TIMEOUT_SECONDS=30
```

如果在内网环境暂时不能访问 Artificial Analysis，可将 `ARTIFICIAL_ANALYSIS_ENABLED=false`，模型追踪的其他功能不受影响。

`QWEN_API_KEY` 为必填项。若使用 OpenAI 兼容的内网 Qwen 服务，可同时修改 `QWEN_BASE_URL` 和 `QWEN_MODEL`；目标服务需要兼容 Chat Completions、Function Calling 与 JSON Object 输出。默认关闭深度思考并使用零温度，以降低费用和输出漂移。

如果使用 Docker Compose，API 与定时任务可以一起启动：

```bash
docker compose up -d --build
```
