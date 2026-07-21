# 手动启动的循环推理方案

这套脚本与 `run_measurement_window.sh` 相互独立，不使用固定钟点。手动启动后按以下节奏循环：

```text
推理 7 小时 → 休息 1 小时 → 推理 7 小时 → 休息 1 小时 → ……
```

手动停止时，脚本会向当前 Python 流程发送 `SIGINT`。vLLM 会先完成并保存当前 chunk，然后退出。2024、2025 的可用数据全部完成后，运行器会自动停止。

## 智能续跑和指定起始阶段

默认的 `start` 现在使用 `auto` 模式，会按每个年份已有的中间文件自动选择起点：

1. 已有 `04_stage2_input_<year>.csv`：直接续跑第二阶段。
2. 已有文本单元、关键词特征，且阶段一结果完整：自动生成缺少的第二阶段路由输入，再续跑第二阶段。
3. 已有文本单元和关键词特征，但阶段一结果不存在或不完整：从阶段一 LLM 初筛续跑。
4. 上述前置结果不完整：才执行完整流程，重新生成文本单元和关键词特征。

阶段一是否完整不是只看“文件存在”，脚本会比较 `01_text_units_<year>.csv` 和 `03_stage1_llm_relevance_<year>.csv` 的 `text_unit_id` 是否完整、一一对应。

通常直接运行即可：

```bash
./deploy/scheduled_llm/manual_cycle_runner.sh start
```

也可以显式指定起始阶段：

```bash
# 复用文本单元和关键词特征，从阶段一初筛断点续跑
./deploy/scheduled_llm/manual_cycle_runner.sh start --from stage1

# 直接从第二阶段续跑；若缺少 04 文件，但阶段一完整，会先自动生成路由输入
./deploy/scheduled_llm/manual_cycle_runner.sh start --from stage2

# 明确要求从头重建文本单元和关键词特征
./deploy/scheduled_llm/manual_cycle_runner.sh start --from full
```

原来的环境变量写法仍然兼容：

```bash
PIPELINE_MODE=stage2 ./deploy/scheduled_llm/manual_cycle_runner.sh start
```

阶段一和阶段二都会读取已有断点。不要添加 `--no-resume`，也不要删除原来的结果 CSV 和 processed log。第二阶段全部完成后，脚本会自动执行 GB 映射、企业年度聚合并创建完成标记。

## 重要：先处理原来的 cron

两套脚本共用同一个文件锁，不会同时占用 GPU；但如果准备长期使用手动循环方案，建议先注释或删除原来每天 23:00 的 cron 行：

```bash
crontab -e
```

否则当手动运行器停止后，原 cron 仍可能在下一次 23:00 自动启动旧方案。

## 部署与启动

```bash
cd /home/suati/桌面/国际标准
chmod +x deploy/scheduled_llm/manual_cycle_runner.sh
bash -n deploy/scheduled_llm/manual_cycle_runner.sh

./deploy/scheduled_llm/manual_cycle_runner.sh start
```

`start` 会在后台运行，所以关闭 SSH、VS Code 或 tmux 不会终止任务。

## 查看状态和日志

```bash
./deploy/scheduled_llm/manual_cycle_runner.sh status
./deploy/scheduled_llm/manual_cycle_runner.sh logs
```

具体年份的模型日志位于：

```text
logs/manual_cycle/main_regression_2024_YYYY-MM-DD.log
logs/manual_cycle/main_regression_2025_YYYY-MM-DD.log
```

## 手动停止

```bash
./deploy/scheduled_llm/manual_cycle_runner.sh stop
./deploy/scheduled_llm/manual_cycle_runner.sh status
```

`stop` 发出安全停止请求后可能需要等待当前 vLLM chunk 完成。再次执行 `status`，看到 `Runner: inactive` 即表示完全退出。

## 修改连续运行时间和休息时间

默认值位于脚本开头：

```bash
RUN_HOURS="${RUN_HOURS:-7}"
REST_HOURS="${REST_HOURS:-1}"
```

可以直接编辑这两个默认值，也可以在每次启动时临时覆盖。例如连续运行 6 小时、休息 2 小时：

```bash
RUN_HOURS=6 REST_HOURS=2 ./deploy/scheduled_llm/manual_cycle_runner.sh start
```

修改节奏前应先执行 `stop` 并确认 `Runner: inactive`，再用新参数 `start`。已经运行的后台进程不会读取后续修改。

为便于短时间测试，脚本还支持按秒覆盖：

```bash
RUN_SECONDS=120 REST_SECONDS=30 ./deploy/scheduled_llm/manual_cycle_runner.sh start
```

正式推理建议使用 `RUN_HOURS` 和 `REST_HOURS`。
