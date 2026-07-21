# 独立回测验证报告 — 2026-07-06 09:33

## 目标
从第一性原理出发，不复用训练框架 reward 函数，对所有已保存策略做独立回测验证。

## 方法
- 脚本: `verify_all_strategies.py`
- target_ret = log(open[t+2]/open[t+1])，无前视偏差
- 仓位 = tanh(factor)，应用 MIN_TRADE_EXPOSURE=0.05
- PnL = pos[t] * target_ret[t] - |pos[t]-pos[t-1]| * cost_rate
- cost_rate = 0.0001 (单边)
- 统计: 年化、Sharpe、Sortino、MDD、胜率、多空比、前后一致性、Walk-Forward 4折、成本压力测试(1x~5x)

## 汇总结果

| 策略 | 年化% | Sharpe | MDD% | 多空比 | H1/H2同号 | WF正折 | 2x成本盈利 | 判定 |
|------|-------|--------|------|--------|-----------|--------|-----------|------|
| forex_v1 (旧) | 3.23 | 0.879 | 7.03 | 38/38 | 是 | 3/3 | 是 | **VALID** |
| forex_v2 (新) | 2.85 | 0.684 | 6.88 | 49/49 | 是 | 2/3 | 是 | **VALID** |
| index_v1 (旧) | -1.79 | -1.136 | 9.68 | 11/7 | 是 | 0/3 | 否 | **INVALID** |
| index_v2 (新) | -1.79 | -1.136 | 9.68 | 11/7 | 是 | 0/3 | 否 | **INVALID** |
| metals_comm_v1 (旧) | 125.63 | 3.420 | 22.18 | 49/49 | 是 | 3/3 | 是 | **INVALID** (MDD>20%) |
| metals_comm_v2 (新) | 151.58 | 4.072 | 13.62 | 55/42 | 是 | 3/3 | 是 | **SUSPICIOUS** (MDD>10%) |

## 详细分析

### Forex 组 (8.01年, 49998 bars)
- **forex_v1** (AROON_OSC_25→TS_MIN_10→EMA_20→CS_SCALE→SIGN→CS_SCALE→EMA_5→MOMENTUM_5)
  - 年化3.23%, Sharpe 0.88, MDD 7.03% — 通过FTMO
  - 多空完美均衡(38/38), 前后一致(H1=2.78%, H2=3.67%), WF 3/3正
  - 2x成本仍盈利(1.45%), 3x转负(-0.33%)
  - 品种: EURUSD +2.57%, USDJPY +3.88%
  - **结论: 有效策略，但收益偏低**

- **forex_v2** (DMI_DIFF_14→HL_RANGE→TS_MIN_20→SUB→TS_MIN_20→TS_MEAN_20→CS_SCALE→WMA)
  - 年化2.85%, Sharpe 0.68, MDD 6.88% — 通过FTMO
  - 多空完美均衡(49/49), 前后一致(H1=3.24%, H2=2.46%), WF 2/3正
  - 2x成本仍盈利(1.43%), 3x接近平手(0.02%)
  - 品种: EURUSD几乎不赚钱(+0.35%), USDJPY贡献全部利润(+5.35%)
  - **结论: 有效但脆弱，利润高度依赖USDJPY**

### Index 组 (5.15年, 32134 bars)
- **index_v1 = index_v2** (同一公式, PPO→TS_RANK_10→TS_SUM_5→TS_RANK_10→TS_SUM_20→TS_MEAN_5→CLIP→SQRT)
  - 年化-1.79%, Sharpe -1.14, 亏损策略
  - 81.8%时间空仓(平), 仅10.8%多/7.4%空
  - WF 0/3正, 所有成本场景均亏损
  - 品种: US30 -2.35%, US100 +3.99%, US500 -7.00%
  - **结论: 完全无效，TS_RANK恒正问题导致因子退化**

### Metals_Comm 组 (1.37年, 8546 bars)
- **metals_comm_v1** (CS_ZSCORE_RET20→EMA_20→TS_MAX_10→TANH_SQUASH→EMA_20→AMIHUD_ILLIQ→MAX→MOMENTUM_5)
  - 年化125.63% — 异常高，主因仅1.37年数据+高波动品种
  - MDD 22.18% — 超过FTMO限制
  - XAUUSD亏损(-11.23%), 利润全部来自AAVUSD(+210%)和COCOA(+178%)
  - **结论: 无效，数据量不足+MDD超标**

- **metals_comm_v2** (SAR_DIST→TREND_STRENGTH_50→TS_RANK_10→TS_MAX_20→ROLL_KURT_20→DELTA_5→TS_DECAY_EXP_5→IF_GT)
  - 年化151.58%, MDD 13.62% — 超FTMO
  - 同样数据量不足
  - **结论: 可疑，不可实盘**

## 关键发现

1. **Forex v1 是最佳策略**: 全维度通过验证，但年化仅3.23%
2. **Index 策略完全无效**: TS_RANK恒正导致因子退化，需重新训练
3. **Metals_Comm 不可靠**: 1.37年数据不足以支撑统计显著性
4. **成本敏感性**: Forex策略在3x成本(0.03%)下转负，实际交易需关注滑点和点差
5. **品种集中风险**: forex_v2利润几乎全部来自USDJPY

## 文件
- 验证脚本: `verify_all_strategies.py`
- 详细结果: `verification_results.json`
- 策略文件: `strategies/` 和 `strategies/archive/`
