# Core Correctness V2 使用指南

以下命令均在 AlphaMaster 项目根目录的 Windows PowerShell 中运行，并使用项目虚拟环境：

```powershell
$PY = 'G:\CodexProject\AlphaMaster\.venv\Scripts\python.exe'
```

## 新训练与精确续训

新建一个独立 V2 运行：

```powershell
# 新训练；不会删除或读取旧 V1 checkpoint、strategy 或 history
& $PY train_file.py `
  --data-file 'D:\K线数据\EURUSD_H1.parquet' `
  --from-scratch
```

`--from-scratch` 每次都创建新的 run ID，并保留磁盘上的所有旧产物。它不会从旧策略播种，也不会删除、改名、覆盖或迁移 V1/V2 历史文件。

使用完全相同的数据、训练配置、随机种子和语义版本精确续训：

```powershell
& $PY train_file.py `
  --data-file 'D:\K线数据\EURUSD_H1.parquet'
```

恢复时会验证 symbol、timeframe、数据指纹、训练配置哈希、核心版本和词表版本。若任一项不同，训练会 fail closed；需要明确重新训练时，重新运行带 `--from-scratch` 的命令。系统不会在 checkpoint 不匹配后静默从头开始。

数据不足以覆盖特征/公式 warmup、两根标签前视、全部 walk-forward block 和不可缩减 gap 时，会抛出最低数据量错误并显示 required/actual。此时应提供更多历史 K 线，不能缩小 gap 或绕过检查。

## 定位并检查不可变策略

```powershell
$strategy = Get-ChildItem -LiteralPath '.\strategies' `
  -Filter 'best_v2_EURUSD_H1_*.json' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName

$strategyJson = Get-Content -LiteralPath $strategy -Raw | ConvertFrom-Json
$trainingData = $strategyJson.run_identity.artifact_identity.training_dataset

$strategy
$trainingData.end_time_ns
$trainingData.data_fingerprint
```

策略文件名包含数据哈希与 run ID 的短标识，内部 JSON 才是权威身份。`training_dataset.end_time_ns` 是 UTC 纳秒训练终点，`training_dataset.data_fingerprint` 是规范化 OHLCV 数据哈希。策略还记录实际 walk-forward fold 时间范围、公式 token、可读公式及核心/词表/标签/执行版本。

## 样本内复盘与独立样本外回测

样本内复盘只能使用策略记录的精确训练数据，用于诊断：

```powershell
& $PY run_backtest.py `
  --strategy-file $strategy `
  --data-file 'D:\K线数据\EURUSD_H1.parquet' `
  --mode in_sample_replay
```

独立样本外回测必须显式提供不同的数据，其起点必须严格晚于训练终点：

```powershell
& $PY run_backtest.py `
  --strategy-file $strategy `
  --data-file 'D:\K线数据\EURUSD_H1_oos.parquet' `
  --mode out_of_sample_backtest
```

选择规则：

- 要复核训练期行为，选择 `in_sample_replay` 和精确训练文件。
- 要评估未知未来行情，选择 `out_of_sample_backtest` 和严格更晚、互不重叠的独立文件。
- 训练数据副本、训练子集、包含训练尾部的扩展数据或更早数据都不能标记为样本外。

成功报告显式记录 `mode`、V2 版本、训练/测试 dataset identity、策略指纹和成本。`ledger_reconciliation` 中的 `ledger_net_pnl`、`execution_net_pnl`、`absolute_difference`、`tolerance` 与 `reconciled` 用于核对逐笔流水和共享执行净收益；流水最后一项包含最终平仓成本。

## 时间与标签语义

对收盘 K 线索引 `t`：

- 因子在 `t` 收盘后形成，仓位在 `t+1` 开盘执行并持有至 `t+2` 开盘。
- `target_ret[t] = log(open[t+2] / open[t+1])`。
- 最后两根 K 线没有完整目标，`target_valid[t]` 为 `false`；它们不会参加 IC、奖励、指标或回测收益。

## V1 与其它训练入口

V1 checkpoint、策略和词表与 V2 不兼容，必须用原始数据重新训练。旧字节会保留在原处，但 V2 不会加载、重标或迁移它们。

`train_single.py` 仍是 MT5/cache 的单标的训练入口，并委托同一 V2 训练服务。group、cross-section 和 island 训练在 V2 中明确禁用；它们不能生成或包装成 V2 策略。
