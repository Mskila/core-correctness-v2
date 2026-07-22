# 历史实验脚本

本目录中的脚本是迁移前的只读研究快照，主要面向旧 `best_*.json`、group 训练、
V1 checkpoint 或特定开发机路径。它们不是 Core Correctness V2 正式入口，不保证
可在当前依赖和产物协议下执行，也不得用于生成、恢复或发布 V2 结果。

当前正式入口：

- Web：`python run_web.py --port 8765`
- 单文件训练：`python train_file.py --data-file <parquet>`
- MT5/cache 单标训练：`python train_single.py <symbol> [--offline]`
- 显式 replay/OOS 回测：`python run_backtest.py ...`

保留这些快照是为了追溯历史分析，不表示继续支持旧分组、旧公式或自动迁移。
