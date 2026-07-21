# 远程设备每日定时推理方案

## 结论

- 2024 可以接着 2015—2023 运行；它使用独立目录 `data/measurement/2024/main_regression`，不会覆盖往年结果。
- 正式续跑时不要使用 `--no-resume`。该选项会删除第二阶段已有结果和处理日志并从头计算。
- `run_measurement_window.sh` 每天只处理 2024、2025：00:00 由 cron 启动，09:55 发送 `SIGINT`，最迟 10:00 强制结束。
- 第一阶段和第二阶段都按 vLLM chunk 保存进度。当天未完成时，次日从未处理的 `text_unit_id` 继续。
- 2025 的目录尚不存在时会自动跳过；以后迁入形如 `年报/2025_XXXX份` 的目录，无需修改脚本。
- 脚本会主动进入项目目录，并向流水线传入配置和项目根目录的绝对路径，避免 cron 从用户主目录启动时找不到 `configs/pipeline.toml`。

## 更改运行时间

脚本默认窗口是 `00:00-10:00`。若改为每天 `23:00-次日10:00`，把 crontab 中原来的任务行替换为：

```cron
0 23 * * * WINDOW_START=23:00 WINDOW_END=10:00 /home/suati/桌面/国际标准/deploy/scheduled_llm/run_measurement_window.sh >> /home/suati/桌面/国际标准/logs/scheduler/cron.log 2>&1
```

编辑方法：

```bash
crontab -e
```

其中 cron 行最前面的 `0 23` 表示每天 23:00 启动；`WINDOW_START` 和 `WINDOW_END` 是脚本内部的安全窗口，二者应保持一致。该脚本支持跨午夜窗口。

默认会在结束前 5 分钟发送安全中断，并在 `WINDOW_END` 强制结束。因此 `23:00-10:00` 实际是 09:55 安全中断、10:00 最终边界。若希望提前 10 分钟安全中断，可增加：

```cron
GRACE_MINUTES=10
```

例如：

```cron
0 23 * * * WINDOW_START=23:00 WINDOW_END=10:00 GRACE_MINUTES=10 /home/suati/桌面/国际标准/deploy/scheduled_llm/run_measurement_window.sh >> /home/suati/桌面/国际标准/logs/scheduler/cron.log 2>&1
```

修改后可检查：

```bash
crontab -l
bash -n /home/suati/桌面/国际标准/deploy/scheduled_llm/run_measurement_window.sh
```

## 一次性部署

把本次修改过的项目文件同步到远程目录后执行：

```bash
cd /home/suati/桌面/国际标准
chmod +x deploy/scheduled_llm/run_measurement_window.sh
mkdir -p logs/scheduler

# 先做语法检查；不会启动模型
bash -n deploy/scheduled_llm/run_measurement_window.sh

# 安装每天 00:00 的任务（保留 crontab 中原有内容）
(crontab -l 2>/dev/null; \
  echo '0 0 * * * /home/suati/桌面/国际标准/deploy/scheduled_llm/run_measurement_window.sh >> /home/suati/桌面/国际标准/logs/scheduler/cron.log 2>&1') \
  | awk '!seen[$0]++' | crontab -
```

确认安装：

```bash
crontab -l
```

不需要让 VS Code SSH 或 tmux 一直在线。cron 由远程 Linux 自己启动任务。`gb` 会话可以只用于观察：

```bash
tmux attach -t gb
tail -F /home/suati/桌面/国际标准/logs/scheduler/main_regression_2024_$(date +%F).log
```

## 首次正式启动前检查

```bash
cd /home/suati/桌面/国际标准
nvidia-smi
find 年报 -mindepth 1 -maxdepth 1 -type d | sort
python3 scripts/run_pipeline.py main-regression --help | grep resume
```

如果 2024 从未开始过，可在 00:00—09:55 内手动执行一次脚本验证；脚本仍会遵守当天 09:55/10:00 边界：

```bash
./deploy/scheduled_llm/run_measurement_window.sh
```

不要让手工命令和 cron 同时运行。脚本中的 `flock` 会阻止两个定时实例重叠，但无法识别另一个未使用该锁的手工 Python 命令。

## 查看状态

```bash
# 调度总日志
tail -n 100 logs/scheduler/cron.log

# 当日 2024 日志
tail -n 100 "logs/scheduler/main_regression_2024_$(date +%F).log"

# 第二阶段已完成任务数
wc -l data/measurement/2024/main_regression/logs/05_stage2_processed_2024_vllm_batch.log

# 完整成功标记
ls -l data/measurement/2024/main_regression/.pipeline_complete
```

`.pipeline_complete` 只有在整条流程正常返回后才创建。若修改了 2024 原始数据、提示词、词库或模型，旧结果不应续用；请先备份或改用新的 `--base-dir`，不要简单删除完成标记后混跑不同版本。

## 停用或恢复

编辑定时任务：

```bash
crontab -e
```

删除或在行首加 `#` 即可停用。恢复时去掉 `#`。若白天需要立刻停止当前任务：

```bash
pkill -INT -f 'scripts/run_pipeline.py main-regression'
```

中断最多损失正在计算、尚未写入检查点的一个 chunk；已落盘 chunk 会在下次运行时跳过。
