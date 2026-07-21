# AlphaMaster V2 策略目录

这里保存训练生成的不可变 `best_v2_*.json` 策略。策略文件包含运行身份、数据
身份、公式词表版本、执行语义版本和指纹；训练、回测与实时信号入口只接受这些
字段完整且身份匹配的 V2 产物。

旧的 `best_*.json`、group 策略、V1 token 列表和无身份字典仍可留在本地作为历史
记录，但会被明确拒绝加载。系统不会删除、重命名、重标或静默迁移它们；需要用
原始数据重新训练生成新的 V2 策略。

策略 JSON 默认被 `.gitignore` 排除，不作为源码提交。正式使用方法见
[`docs/core-correctness-v2-usage.md`](../docs/core-correctness-v2-usage.md)。历史 V1
公式、验证报告和实验脚本已经归档在 [`extras/`](../extras/README.md)，不代表当前
能力或可信绩效基线。
