# AlphaMaster

基于深度神经网络强化学习的量化因子挖掘中心：从 Parquet / MT5 K 线自动搜索可解释因子公式，支持 Web 端训练、回测与实时信号分析。

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

![Web 控制台总览](docs/images/00_hero.png)

仓库地址：[github.com/Mskila/core-correctness-v2](https://github.com/Mskila/core-correctness-v2)

---

## Core Correctness V2

V2 使用严格因果特征、精确 checkpoint 身份、不可变策略，以及显式区分的样本内复盘和独立样本外回测。命令行快速开始、时间语义、续训规则和报告字段见 [Core Correctness V2 使用指南](docs/core-correctness-v2-usage.md)。

> **重要：旧 checkpoint 和策略与 V2 不兼容，必须使用原始数据重新训练。旧产物会保留，但不会被加载、覆盖或迁移。**

---

## 它做什么

AlphaMaster 把「挖因子」做成一条可操作的流水线：

1. **训练**：用强化学习在特征 + 算子空间里搜索公式，按验证集表现选优  
2. **回测**：用 `tanh(因子)` 连续仓位在历史行情上模拟交易，看资金曲线与绩效  
3. **实时分析**：按周期收盘后重算信号，展示方向与把握；方向转折可推飞书提醒  

公式以 token 序列保存为带数据与运行身份的不可变 V2 策略（如 `strategies/best_v2_BTCUSDT_H1_<identity>.json`），可用 StackVM 解释执行，训练 / 回测 / 实时共用同一套信号逻辑。

---

## Web 控制台（推荐入口）

```bash
pip install -r requirements.txt
python run_web.py --port 8765
```

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。界面分三步：

| 步骤 | 作用 |
|------|------|
| **01 模型训练** | 选 Parquet、开始 / 重新训练、看曲线与日志、导出策略与检查点 |
| **02 策略回测** | 选策略 JSON，设手续费 / 滑点，看绩效与资金曲线 |
| **03 实时分析** | 多数据源监控，收盘后更新信号；可选飞书转折提醒 |

### 模型训练

![训练页](docs/images/01_train.png)

- Parquet 命名：`{品种}_{周期}.parquet`，例如 `BTCUSDT_H1.parquet`、`XAUUSD_H1.parquet`  
- **开始训练**：仅在数据、配置与版本身份完全一致时从 V2 检查点续训；否则失败并要求显式重新训练
- **重新训练**：`from_scratch` 创建全新运行并保留旧产物，不读取或删除 V1 产物，也不沿用旧策略分数下限
- 展示最优分数、验证分数、训练曲线与最优公式；可选 AI 分析当前训练情况  

### 策略回测

![回测页](docs/images/02_backtest.png)

- 仓位：`position = tanh(factor)`，信号越强仓位越大  
- 成本：手续费 + 滑点（默认约 0.02% / 0.01%）  
- 输出：总收益、夏普、索提诺、盈亏比、滚动夏普与资金曲线  

![资金曲线示例](docs/images/04_equity.png)

### 实时分析

![实时分析页](docs/images/03_realtime.png)

- 数据源：MT5 / OKX 等（以界面可用源为准）  
- **只在当前周期 K 线收盘后**重新判断；未收盘 bar 不参与信号  
- 卡片展示方向（看涨 / 看跌 / 不确定）与把握程度  
- 可选飞书 Webhook：仅在方向转折时推送文字提醒  

---

## 项目结构

```
AlphaMaster/
├── web/                 # FastAPI Web UI（训练 / 回测 / 实时）
├── model_core/          # 特征、算子、StackVM、训练引擎、回测评分
├── data_pipeline/       # Parquet / MT5 K 线加载与对齐
├── strategy_manager/    # 实盘信号与仓位逻辑（与回测口径一致）
├── execution/           # MT5 下单接口
├── backtest_viz/        # 回测引擎与图表
├── strategies/          # 不可变、带身份的 best_v2_* 策略文件
├── extras/              # 非核心实验脚本与历史记录（不属于 V2 正式入口）
├── checkpoints/         # 训练检查点
├── run_web.py           # 启动 Web 控制台
├── train_file.py        # CLI：从单个 Parquet 训练
└── requirements.txt
```

---

## 环境要求

- Python **3.10+**（建议 3.11）  
- PyTorch、pandas、FastAPI、uvicorn 等（见 `requirements.txt`）  
- 可选：MetaTrader 5 终端（实时 MT5 源 / 实盘相关脚本）  
- 复制 `.env.example` 为 `.env` 填写 MT5 等凭证（`.env` 已 gitignore）  

```bash
pip install -r requirements.txt
# 可选可视化等：pip install -r requirements-optional.txt
```

---

## 常用命令

```bash
# Web 控制台
python run_web.py --port 8765

# CLI 训练（身份完全一致时续训；加 --from-scratch 创建新运行并保留旧产物）
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet
python train_file.py --data-file D:\K线数据\BTCUSDT_H1.parquet --from-scratch
```

策略输出为 `strategies/best_v2_<symbol>_<timeframe>_<identity>.json` 形式的不可变 V2 文件。

正式入口只有本节和 V2 使用指南列出的命令。根目录不再放置旧 group、V1
回测或临时 checkpoint 分析入口；这些历史快照保留在 `extras/`，不参与 V2
训练、回测、发布门禁或依赖安装。

---

## 信号口径（训练 / 回测 / 实时一致）

- 因子经 StackVM 算出标量序列  
- `position = tanh(factor)` ∈ (-1, 1)  
- `|position|` 小于阈值时视为无信号（观望）  
- 实时侧只用**已收盘** K 线，避免盘中抖动与回测不一致  

---

## 截图更新

仓库内展示图由当前 Web UI 截取，可用：

```bash
python scripts/capture_readme_shots.py
```

（需本机已启动 `python run_web.py --port 8765`，并已安装 Playwright + Chromium。）

---

## License

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)。  
修改、分发或通过网络提供服务时，须按相同协议公开对应源代码。

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Mskila/core-correctness-v2&type=date&legend=top-left)](https://www.star-history.com/#Mskila/core-correctness-v2&type=date&legend=top-left)
