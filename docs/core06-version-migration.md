# CORE-06 版本边界与重新训练说明

CORE-06 是一次严格的语义边界升级。当前正式产物使用 `checkpoint-v3`、
`strategy-v3`、`training-history-v3` 和 `backtest-report-v3`。核心、词表、
执行、数据规范化与验证版本的权威对应关系由
`model_core.versions.VERSION_CHANGE_TABLE` 提供。

## 旧产物处理

旧 checkpoint、strategy、history 和 report 均保留原文件，不删除、不改名、
不覆盖，也不静默迁移。它们统一视为 `pre-core-fix/incompatible`：

- 不能恢复训练；
- 不能进入候选排名；
- 不能执行正式回测或生成正式报告；
- 不能作为当前 Web 训练进度或当前策略展示。

`model_core.legacy_audit.inspect_legacy_artifact` 只读取 JSON 基本元数据，返回的
结果明确标记 `rank_eligible=False` 和 `formal_output_allowed=False`。该工具不会
写入、迁移或包装旧产物。

## 重新训练

使用原始训练数据并显式选择从头训练。每次从头训练都会创建新的 run ID 和
新的不可变文件名，因此不会覆盖旧产物。只有数据身份、训练配置、随机种子及
全部语义版本完全一致时，才允许从当前 v3 checkpoint 精确恢复；任一不一致都
会明确拒绝，且不会静默退回从头训练。

Web 和旁路读取器只把当前 v3 产物计入正式状态。磁盘上的旧文件仍可由人工或
只读审计工具查看，但会显示为不兼容/修复前产物。
