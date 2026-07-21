# 国际标准识别流程

这个项目把早期 notebook 流程整理为可复用的国际标准测度流水线。当前包含主回归、三种稳健性检验和跨方法一致性比较；同时保留原有的三段式入口：

1. `preprocess`：从年报 txt 中抽取包含标准、认证、体系关键词的片段。
2. `extract`：调用 API 或本地模型，将片段抽取为结构化标准实体。
3. `map-gb`：把 `TYPE_B` 的 GB/行业标准映射到国际标准采标信息，并生成最终 `output` 标记。

## 安装

项目要求 Python 3.11 或更高版本。`pyproject.toml` 是唯一的依赖配置来源，所有依赖变更都应写入其中。

基础安装（支持数据处理和 OpenAI-compatible API）：

```powershell
python -m pip install -e .
```

如果使用 Transformers 本地模型：

```powershell
python -m pip install -e ".[local-model]"
```

如果启用 4-bit 量化，在本地模型依赖之外安装量化依赖：

```powershell
python -m pip install -e ".[local-model,quantization]"
```

开发和测试环境：

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

安装后可使用控制台命令：

```powershell
standard-pipeline --help
```

也可以直接使用仓库内入口：

```powershell
python scripts/run_pipeline.py --help
```

## 配置

主配置文件在 `configs/pipeline.toml`。默认沿用当前目录下的数据结构：

- 年报原文：`年报/`
- 预处理输出：`预处理数据/`
- 最终预测输出：`最终预测数据/`
- GB 映射库：`GB映射参考数据库/GB_dict.csv`
- 关键词表：`configs/standard_keywords.txt`
- 抽取提示词：`prompts/extract_standards_zh.txt`

API 密钥不要写进代码。运行前在 PowerShell 中设置环境变量：

```powershell
$env:STANDARD_PIPELINE_API_KEY="你的密钥"
$env:STANDARD_PIPELINE_BASE_URL="https://api.deepseek.com"
$env:STANDARD_PIPELINE_MODEL="deepseek-chat"
```

## 常用命令

新的测度流水线按方法分目录管理，默认输出到：

- `data/measurement/main_regression/`
- `data/measurement/robustness_keyword/`
- `data/measurement/robustness_llm_only/`
- `data/measurement/robustness_full_llm/`
- `data/measurement/comparison/`

各方法的实现统一放在 `src/standard_pipeline/`，由 `scripts/run_pipeline.py` 或安装后的 `standard-pipeline` 命令调用。每个方法的数据目录固定使用 `stage/`、`results/`、`final/`、`logs/`，输出文件按处理顺序编号。加 `--limit` 的冒烟测试会写入 `data/measurement_smoke/<method>/`，不和正式结果混用。

主回归：

```powershell
python scripts/run_pipeline.py main-regression --year 2024 --provider api
```

在重建正式文本单元前，可先对现有 `01_text_units_<year>.csv` 做非覆盖式清洗审计：

```powershell
python scripts/run_pipeline.py audit-text-units --year 2021 `
  --input "data/measurement/2021/main_regression/stage/01_text_units_2021.csv" `
  --output "data/measurement/2021/main_regression_audit/stage/01_text_units_2021_audit.csv" `
  --base-dir "data/measurement/2021/main_regression_audit"
```

该命令拒绝将审计结果写回输入路径，并输出页眉页码及表格行压缩前后的长度、删除比例、保护词命中和预览。正式重建时，章节标题仅按独立行识别；重复章节页眉不会新建章节边界，股票代码统一保留为六位字符串。

只抽样构建文本单元（不运行关键词特征和 LLM）：

```powershell
# 每个年份随机抽 10 份，固定 seed 后可重复得到同一批样本
python scripts/sample_text_units.py --years 2019 2020 2021 --sample-size 10 --seed 20260718

# 定向检查指定股票代码；0 表示处理全部匹配报告
python scripts/sample_text_units.py --years 2020 2021 2022 `
  --stock-codes 000001 000002 --sample-size 0 --overwrite
```

默认输出到 `data/text_unit_samples/<year>/01_text_units_<year>_sample.csv`。脚本只执行 TXT 章节识别、页眉页码清理、表格行压缩和文本单元切分；不会生成关键词文件，也不会调用任何模型。

稳健性检验：

```powershell
python scripts/run_pipeline.py robustness-keyword --year 2024 --provider api
python scripts/run_pipeline.py robustness-llm-only --year 2024 --provider api
python scripts/run_pipeline.py robustness-full-llm --year 2024 --provider api
```

横向一致性比较：

```powershell
python scripts/run_pipeline.py compare-measurements --year 2024
```

四份企业-年度结果分别为：

- `data/measurement/main_regression/final/07_main_regression_firm_year_<year>.csv`
- `data/measurement/robustness_keyword/final/04_robustness_keyword_firm_year_<year>.csv`
- `data/measurement/robustness_llm_only/final/04_robustness_llm_only_firm_year_<year>.csv`
- `data/measurement/robustness_full_llm/final/04_robustness_full_llm_firm_year_<year>.csv`

只做年报预处理：

```powershell
python scripts/run_pipeline.py preprocess --year 2024
```

用 API 模型抽取：

```powershell
python scripts/run_pipeline.py extract --input "预处理数据/yuchuli_2024.csv" --provider api
```

用本地模型抽取：

```powershell
$env:LOCAL_MODEL_PATH="A:\models\qwen3.6-27B"
python scripts/run_pipeline.py extract --input "预处理数据/yuchuli_2024.csv" --provider local
```

### provider=vllm_batch

`provider=vllm_batch` 会在当前 pipeline Python 进程中直接加载 `vllm.LLM`，并使用 `llm.generate(prompts, sampling_params)` 对多条 prompt 做批量推理。这个模式不是连接已经启动的 OpenAI-compatible `vllm serve` 服务。

请在支持 vLLM 的 Linux/NVIDIA GPU 环境中安装对应可选依赖：

```bash
python -m pip install -e ".[vllm]"
```

使用前建议先检查 GPU：

```powershell
nvidia-smi
```

如果同一张 GPU 上已经有 `vllm serve` 或其他大显存进程，建议先停止，否则 `vllm_batch` 在当前进程里再次加载模型时容易 OOM。

示例：

```powershell
python scripts/run_pipeline.py stage1-screen --year 2024 --provider vllm_batch
python scripts/run_pipeline.py stage2-extract --year 2024 --provider vllm_batch
python scripts/run_pipeline.py main-regression --year 2024 --provider vllm_batch
```

执行 GB 映射：

```powershell
python scripts/run_pipeline.py map-gb --input "最终预测数据/final_2024.csv"
```

完整链路：

```powershell
python scripts/run_pipeline.py run-all --year 2024 --provider api
```

冒烟测试可以加 `--limit`：

```powershell
python scripts/run_pipeline.py run-all --year 2024 --provider api --limit 5
```

## 输出字段

`extract` 输出字段：

- `stock_code`
- `company_name`
- `year`
- `entity`
- `type`
- `status`
- `evidence`

`map-gb` 在此基础上新增：

- `国际标准`
- `采标情况`
- `output`

其中 `output=1` 表示企业已采纳国际标准或经 GB 映射确认采用国际标准；否则为 `0`。
