# 回测前视偏差审计报告

**时间**: 2026-07-06 00:10
**审计范围**: backtest_all_groups.py, backtest_current.py, backtest_detailed.py, backtest_elite.py, backtest_index_best.py, backtest_current_forex.py
**结论**: 主力回测脚本无前视偏差；backtest_current_forex.py 有 target_ret 构造错误

---

## 1. Target Return 构造

### 训练引擎 (engine.py)
- 来源: `data_manager.target_ret`
- 公式: `target_ret[n, t] = log(open[t+2] / open[t+1])`
- 含义: t 时刻决策 → t+1 开仓 → t+2 平仓的收益
- **正确，无前视**

### 主力回测脚本 (backtest_all_groups.py, backtest_current.py, backtest_detailed.py, backtest_elite.py, backtest_index_best.py)
- 来源: `group_mgr.target_ret` / `mgr.target_ret`
- 与训练引擎一致
- **正确，无前视**

### 临时脚本 (backtest_current_forex.py) ❌
- 来源: 手动构造 `target_ret[:, 1:] = (close[:, 1:] / close[:, :-1] - 1)`
- 公式: `target_ret[t] = close[t]/close[t-1] - 1` (t-1→t 的收益放在 t 位置)
- 问题: `pnl[t] = pos[t] * (close[t]/close[t-1] - 1)` 相当于用 t 时刻信息决定 t-1 的仓位
- **有前视偏差**，但此脚本仅为临时检查用途，不影响正式回测结论

---

## 2. PnL 计算时序

### 训练引擎 evaluate_fold
```python
position = compute_target_positions_stateless(factors)  # pos[t] = tanh(factor[t])
pnl = position * target_ret - turnover * cost_rate
# pnl[t] = tanh(factor[t]) * log(open[t+2]/open[t+1]) - |pos[t]-pos[t-1]| * cost
```

时序:
1. t 时刻: close[t] 完成 (H1 K线 [t, t+1) 收盘) → 计算 factor[t] → 决定 pos[t]
2. t+1 时刻: 以 open[t+1] 建仓 (换手成本在此刻发生)
3. t+2 时刻: 以 open[t+2] 平仓
4. 收益 = log(open[t+2]/open[t+1])

**因果正确**。pos[t] 基于 close[t]（t+1 时刻可知），收益从 t+1 开始。

### backtest_all_groups.py
```python
pos = compute_target_positions_stateless(factor)
pnl = pos_np * target_np - turnover * cost_rate
```
与训练引擎一致。**正确**。

---

## 3. 特征引擎前视检查

### _robust_norm (features.py L137)
- 因果滚动窗口: `[t-w+1..t]` 共 w 期
- unfold + median/MAD
- **无前视**

### SCALE 算子 (ops.py L237)
- `scale(x)[t] = x[t] / cumsum(|x[0..t]|)` (因果累积和)
- **无前视**

### 特征中 close 的使用
所有特征只用 `close[t]` 和历史值 (`close[:, :-1]` 等)，无 `close[t+1]` 或未来索引。
`close[t]` 在 H1 K 线中需 t+1 时刻才完成，但 target_ret 从 t+1 开始计收益，时序对齐正确。

### TS_RANK / TS_MIN / TS_MAX 等滚动算子
均使用 `_ts_rolling(x, d)` = `x.unfold(1, d, 1)`，是因果窗口 [t-d+1..t]。
**无前视**。

---

## 4. Walk-Forward 折叠

```python
folds = _build_walk_forward_folds(T, n_folds, gap=20)
```
- 第 k 折: train[0, train_end) → gap → val[val_start, val_end)
- train_end ≤ val_start - gap，验证段严格在训练段之后
- **无前视**

---

## 5. 换手率与成本

```python
prev[:, 1:] = pos[:, :-1]  # prev[t] = pos[t-1]
turnover = |pos - prev|    # turnover[t] = |pos[t] - pos[t-1]|
```
- t 时刻的仓位变化 (pos[t-1] → pos[t])
- 成本在 t 时刻发生，收益从 t+1 开始
- **因果正确**

---

## 6. compute_target_positions_stateless (signal.py)

```python
pos = torch.tanh(factors)
pos = torch.where(pos.abs() >= 0.05, pos, 0)  # MIN_TRADE_EXPOSURE
```
- 逐元素操作，无时序依赖
- **无前视**

---

## 7. IC 计算说明

backtest_all_groups.py 中:
```python
x = factor[n, :-1]      # factor[t]
y = target_ret[n, 1:]   # target_ret[t+1] = log(open[t+3]/open[t+2])
```
衡量 factor[t] 对 t+2→t+3 收益的预测力（2 步滞后 IC）。
这不是前视，只是 IC 指标的定义方式，不影响 PnL 计算。

---

## 总结

| 检查项 | 状态 |
|--------|------|
| target_ret 构造 | ✅ 正确 (主力脚本) |
| PnL 时序对齐 | ✅ 正确 |
| 特征引擎因果性 | ✅ 正确 |
| _robust_norm 因果性 | ✅ 正确 |
| SCALE 算子因果性 | ✅ 正确 |
| Walk-Forward 折叠 | ✅ 正确 |
| 换手率计算 | ✅ 正确 |
| backtest_current_forex.py | ❌ target_ret 构造错误 (临时脚本，不影响正式结论) |

**最终结论**: 主力回测脚本 (backtest_all_groups.py 等) 无前视偏差，回测结果可信。backtest_current_forex.py 有 target_ret 构造错误但不影响正式回测结论。
