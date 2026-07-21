# 历史仓库速读（V1，非当前能力说明）

> 本文已归档。下述 group 策略、`best_*.json`、固定实盘流程和自动下单描述仅用于
> 追溯旧版本；Core Correctness V2 会拒绝这些无身份产物，且正式入口不依赖本文。

AlphaGPT MT5 仓库速读

**开源协议**：GNU Affero General Public License v3.0 (AGPL-3.0)。修改、分发或通过网络提供服务时，须按相同协议公开源代码。详见根目录 `LICENSE`。

这是一套面向 MetaTrader 5 的「因子挖掘 + 回测 + 实盘执行」量化系统。核心思路是：用 Transformer 自动生成可解释的因子公式，通过回测打分筛选，再把高分公式用于 MT5 实时信号与下单。

代码组织（按功能划分）
- data_pipeline/：MT5 行情拉取、本地 K 线 Parquet 缓存、多品种数据对齐
- model_core/：策略挖掘。特征工程、算子 DSL、StackVM 执行、AlphaGPT 训练与回测评分
- strategy_manager/：实盘策略执行。信号生成、风控、持仓管理
- execution/：MT5 下单与报价（MT5Trader、MT5PriceFeed）
- backtest_viz/：回测图表与报告
- paper/、lord/、times.py：研究材料或独立实验脚本（不直接参与主流程）

主流程（从数据到实盘）
1) data_pipeline 从 MT5 或本地缓存加载 H1 OHLCV
2) model_core 训练生成最优公式（strategies/best_{symbol}.json）
3) strategy_manager 读取公式，计算 tanh 连续仓位信号
4) risk 计算手数，execution 通过 MT5 API 下单
5) portfolio 持仓落地到本地 JSON（portfolio_state.json）

核心思想
- 不是直接预测价格，而是「生成公式 → 解释执行 → 回测评分 → 优化生成器」
- 公式 = token 序列；token 由「特征 + 算子」组成，StackVM 执行成因子信号
- 交易层只消费最终信号分数，负责风控与执行

现状与依赖
- 需要 MetaTrader5 终端、python-dotenv、PyTorch
- MT5 凭证通过 .env 配置（见 .env.example）
- 策略 JSON 需先训练生成，或使用仓库内已有的 strategies/best_*.json
