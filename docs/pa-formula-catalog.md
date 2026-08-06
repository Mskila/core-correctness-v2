# PA 数学公式目录（阶段 1A）

状态：阶段 1A 冻结候选；只定义公式、来源和实现边界，不包含生产代码。

本目录是阶段 1B 的唯一公式合同。阶段 1B 只有在本目录通过验收后才能开始；后续若修改公式、阈值或因果语义，必须先修改本目录并单独验收。

## 1. 来源、许可与范围

- 上游项目：[PA_Agent](https://github.com/rosemarycox5334-debug/PA_Agent)
- 固定来源提交：[`d92ecd827fe671a589b7fdfdbba41e5e98081d87`](https://github.com/rosemarycox5334-debug/PA_Agent/tree/d92ecd827fe671a589b7fdfdbba41e5e98081d87)
- 上游许可：`AGPL-3.0-or-later`，Copyright (C) 2026 PA Agent Contributors。
- AlphaMaster 本身同样采用 AGPL-3.0；阶段 1B 复制、改写或派生上游算法时，模块头和项目归属说明必须保留上述来源、固定提交和许可证，不得把上游代码标成 AlphaMaster 原创。
- 本阶段只提取确定性 K 线/PA 算法与可被确定性形式化的策略结构，不引入 PA_Agent 的 AI 客户端、提示词编排、GUI、通知、数据源或自动下单实现。
- 成交量不是任何公式的必要输入；本目录中的全部判断只依赖已收盘 OHLC、时间、EMA20、ATR14、最小价格跳动 `tick` 与由它们派生的结构。

主要来源文件：

- `pa_agent/ai/kline_features.py`
- `pa_agent/ai/market_features.py`
- `pa_agent/ai/decision_nodes.py`
- `pa_agent/ai/structure_levels.py`
- `pa_agent/indicators/atr.py`
- `pa_agent/indicators/ema.py`
- `prompt_engineering/市场诊断框架.txt`
- `prompt_engineering/二元决策.txt`
- `prompt_engineering/文件13` 至 `文件25`、`文件28` 中与本目录范围对应的规则

三角形、开盘时段策略、新闻/流动性叙事、仓位管理和 PA_Agent 的 AI 输出 schema 不属于已批准的阶段 1A 工作包，不在阶段 1B 实现。

### 1.1 公式到来源的追溯表

| 本目录范围 | 固定提交中的权威来源 |
|---|---|
| 单棒几何、inside/outside、ii/iii/ioi、微双顶底、EMA gap、跟随 | `pa_agent/ai/kline_features.py` |
| ATR14 / EMA20 | `pa_agent/indicators/atr.py`、`pa_agent/indicators/ema.py` |
| 区间位置、Barbwire、`r=1` 枢轴、突破/失败/回测、原始 H/L 累计、MM 候选 | `pa_agent/ai/market_features.py` |
| `r=1` 支撑阻力与最近结构位 | `pa_agent/ai/structure_levels.py` |
| 五信号方向、`r=2` 枢轴、Always-In、动量与信号棒门 | `pa_agent/ai/decision_nodes.py` |
| 通道分类与回撤阈值 | `prompt_engineering/市场诊断框架.txt`、`文件13-窄通道与宽通道策略.txt` |
| H1/H2/L1/L2 与二次入场 | `prompt_engineering/文件19-H1H2-L1L2计数.txt`、`文件15-二次入场机会.txt` |
| 突破、失败、回测、失败的失败 | `prompt_engineering/文件18-突破失败与突破测试.txt` |
| Always-In、20GB 与 EMA gap 语义 | `prompt_engineering/文件20-AlwaysIn与20GB.txt`、`文件16-K线信号识别.txt` |
| Barbwire/no-trade 环境 | `prompt_engineering/文件21-铁丝网与无交易环境.txt` |
| wedge / MTR / Final Flag / 双顶底 | `prompt_engineering/文件14-楔形形态分析交易.txt`、`文件25-主要趋势反转MTR.txt`、`文件24-最终旗形与趋势末端.txt`、`文件28-双重顶底与微型结构.txt` |
| 测量目标与结构目标 | `prompt_engineering/文件23-MeasuredMove与结构目标.txt`、`文件22-信号失败后的磁力位.txt` |
| 趋势延续、突破、反转、区间四类策略语义 | `极速上涨/下跌交易策略.txt`、`上涨/下跌通道交易策略.txt`、`震荡区间交易策略.txt`、`二元决策.txt` |

## 2. 规范输入与因果约束

### 2.1 时间方向

AlphaMaster 内部只使用时间升序：

```text
B[0], B[1], ..., B[t]
```

`B[t]` 是决策时点可见的最新已收盘 K 线。上游 PA_Agent 使用 newest-first，映射固定为：

```text
K1 = B[t]
Kj = B[t-j+1]
```

适配器必须先删除正在形成的 K 线，再做方向转换。任何公式在 `as_of=t` 时只能读取 `B[0:t+1]`；不得使用 `B[t+1:]`，也不得把后来确认的枢轴结果回填为在枢轴当根已经可见。

### 2.2 K 线与记号

对已收盘 K 线 `B[i]`：

```text
O_i = open
H_i = high
L_i = low
C_i = close
R_i = H_i - L_i
body_i = abs(C_i - O_i)
eps = 数值稳定用极小正数，不作为价格容差
tick = 当前品种最小价格跳动
```

输入必须是有限数值并满足 `H_i >= max(O_i, C_i)`、`L_i <= min(O_i, C_i)`。公式层不修复坏 OHLC；无效输入返回 `valid=false`。

### 2.3 数据段、预热和确认时间

- EMA、ATR、形态窗口和波段状态不得跨越数据缺口分段；新分段重新预热。
- EMA20 在第 20 根有效同段 K 线前无值；ATR14 在第 14 根有效同段 K 线前无值。
- 需要右侧 `r` 根确认的枢轴 `i`，最早在 `i+r` 收盘后发布，输出 `confirmed_at=i+r`。回测与实盘均按确认时间使用。
- 分母为零、预热不足、窗口跨段或必要枢轴未确认时，返回无效/未知，不把缺失值静默当成零、假或中性。

### 2.4 输出共同字段

每个阶段 1B 公式结果至少携带：

```text
formula_id, value, valid, as_of, source_start, source_end,
confirmed_at, invalid_reason, provenance_class
```

价格输出保持原价格单位；比例输出在 `[0,1]`；有向分数在 `[-1,1]`；离散结果使用本文列出的枚举。

## 3. 来源等级与优先级

| 等级 | 含义 | 阶段 1B 处理 |
|---|---|---|
| `D0` | 固定提交中已有确定性实现，语义与实现一致 | 按上游公式移植，并用 golden fixture 锁定 |
| `D0-C` | 上游已有确定性实现，但存在时间方向、命名或语义漂移 | 按本目录的修正规则实现，不复制缺陷 |
| `D1` | 上游只在 playbook/提示词中描述，由本目录给出确定性形式化 | 以本目录公式为准，测试必须覆盖阈值边界 |
| `P` | 上游执行政策，不是市场公式 | 不进入阶段 1B 公式输出；在阶段 2/5 单独处理 |

发生冲突时，优先级为：本目录的因果合同与显式修正规则 > 固定提交的确定性代码 > 固定提交的 playbook 描述 > 概率性文字叙事。

## 4. 单根 K 线与基础指标

### 4.1 几何公式

| ID | 定义 | 输出/范围 | 窗口 | 无效条件 | 来源 |
|---|---|---|---|---|---|
| `bar.range` | `R_i = H_i - L_i` | 价格，`>=0` | 1 | OHLC 无效 | `D0` |
| `bar.body` | `abs(C_i-O_i)` | 价格，`>=0` | 1 | OHLC 无效 | `D0` |
| `bar.body_ratio` | `body_i/R_i` | `[0,1]` | 1 | `R_i=0` | `D0` |
| `bar.upper_wick_ratio` | `(H_i-max(O_i,C_i))/R_i` | `[0,1]` | 1 | `R_i=0` | `D0` |
| `bar.lower_wick_ratio` | `(min(O_i,C_i)-L_i)/R_i` | `[0,1]` | 1 | `R_i=0` | `D0` |
| `bar.close_position` | `(C_i-L_i)/R_i`，再裁剪到 `[0,1]` | `[0,1]` | 1 | `R_i=0` | `D0` |
| `bar.direction` | `bull` if `C_i>O_i`; `bear` if `<`; else `flat` | 枚举 | 1 | OHLC 无效 | `D0` |
| `bar.range_atr_ratio` | `R_i/ATR14_i` | `[0,+inf)` | 1+ATR | ATR 未预热或 `<=0` | `D0` |

K 线类型按以下互斥顺序裁定，顺序本身是合同：

1. `inside`：`H_i <= H_{i-1}` 且 `L_i >= L_{i-1}`。
2. `outside_bull/outside_bear`：`H_i >= H_{i-1}` 且 `L_i <= L_{i-1}`，再按 `C_i >= O_i` 选 bull，否则 bear。
3. `flat`：`R_i=0`。
4. `doji`：`body_ratio_i <= 0.25`。
5. `trend_bull`：`C_i>O_i` 且 `close_position_i >= 0.65`。
6. `trend_bear`：`C_i<O_i` 且 `close_position_i <= 0.35`。
7. 其他为 `other`。

这套类型来自 `kline_features._classify_bar`，等级 `D0`。playbook 中“实体大于近期均值 1.5 倍”的说法属于信号质量增强项，不替换上述基础类型。

### 4.2 ATR14 与 EMA20

True Range：

```text
TR_0 = H_0 - L_0
TR_i = max(H_i-L_i, abs(H_i-C_{i-1}), abs(L_i-C_{i-1}))
```

ATR14 使用 Wilder 平滑：

```text
ATR14_13 = mean(TR_0 ... TR_13)
ATR14_i = (13 * ATR14_{i-1} + TR_i) / 14, i >= 14
```

EMA20 使用前 20 个收盘的简单平均作种子：

```text
EMA20_19 = mean(C_0 ... C_19)
alpha = 2 / 21
EMA20_i = alpha*C_i + (1-alpha)*EMA20_{i-1}, i >= 20
```

两者均为 `D0`。阶段 1B 必须证明整段计算与逐根增量计算逐点一致。

### 4.3 相邻关系与收缩组合

相邻 K 线重叠：

```text
shared = max(0, min(H_i,H_{i-1}) - max(L_i,L_{i-1}))
union  = max(H_i,H_{i-1}) - min(L_i,L_{i-1})
overlap_i = shared / union
```

`union=0` 时无效。输出 `[0,1]`，等级 `D0`。

组合定义：

- `ii`：`inside(i,i-1)` 且 `inside(i-1,i-2)`。
- `iii`：在 `ii` 基础上再满足 `inside(i-2,i-3)`。
- `ioi`（时间升序）：`inside(i-2,i-3)`、`outside(i-1,i-2)`、`inside(i,i-1)`。
- `breakout_prev_5=up`：`H_i > max(H_{i-5:i})`；`down` 对称；两边同时突破为 `both`。

均为 `D0`。窗口不足时返回 `none/valid=false`，不能用缩短窗口冒充完整形态。

### 4.4 跟随、微双顶底与 EMA 侧别

对已完成的多头信号棒 `s`，只看其后最多两根已收盘 K 线：

```text
follow=yes    if any C_j > C_s
follow=failed if no C_j > C_s and any C_j < O_s
follow=no     otherwise
```

空头对称：`yes` 使用 `C_j<C_s`，`failed` 使用 `C_j>O_s`。没有后续已收盘棒为 `pending`。当 `yes` 与 `failed` 证据同时出现时，上游实现优先返回 `yes`；阶段 1B 保持该顺序，等级 `D0`。

上游微型双结构：

```text
tol_micro = 0.02 * ATR14_i                  (ATR 有效)
tol_micro = 0                               (ATR 无效，仅精确相等)
MDB if abs(L_i-L_{i-1}) <= tol_micro
MDT if abs(H_i-H_{i-1}) <= tol_micro
```

若同时满足，按上游顺序优先 `MDB`。这是 `D0` 原始特征；完整双顶/底另见 7.4。

EMA 关系：

```text
ema_relation = above if C_i>EMA20_i, below if C_i<EMA20_i, touch otherwise
ema_gap_above if L_i>EMA20_i
ema_gap_below if H_i<EMA20_i
```

上游把这两个几何状态命名为 `bull_gap`/`bear_gap`，而 playbook 又把“均线另一侧 gap bar”按市场背景叙述，容易混义。阶段 1B 规范字段只用 `ema_gap_above/ema_gap_below`，兼容层才可映射旧枚举，等级 `D0-C`。

`ema_gap_run_i` 是截至 `i`、整根 K 线连续处于 EMA 同一侧的长度；一旦侧别变化、触及 EMA、EMA 无值或跨数据段即归零。`ema_gap_run_i >= 20` 产生 `twenty_gap_bars=true`，不等价于趋势结束，等级 `D0-C`。

相邻 K 线之间的真实价格缺口另行输出：

```text
interbar_gap_up   if L_i > H_{i-1}+tick
interbar_gap_down if H_i < L_{i-1}-tick
```

这是从 playbook 的 opening-gap 叙述中抽出的 `D1` 几何事实，不和 EMA gap 混用。`opening_gap` 还要求权威交易日/session 边界；阶段 1B 没有规范 session calendar 时不得仅按本机日期猜测，只输出 `interbar_gap_*`。

## 5. 区间、波段与方向

### 5.1 区间包络与位置

默认结构窗口 `W=40`：

```text
range_high_t = max(H_{t-W+1:t+1})
range_low_t  = min(L_{t-W+1:t+1})
range_width  = range_high_t - range_low_t
price_position = (C_t-range_low_t)/range_width
```

- `<1/3` 为 `lower_third`；`>2/3` 为 `upper_third`；其余为 `middle_third`。
- `range_width_atr = range_width/ATR14_t`。
- 到上下沿距离也以 `ATR14_t` 归一化。

窗口不足 40 时可使用当前同段可用窗口，但必须在结果中记录实际 `lookback_bars`；不足 3 根时结构结果无效。等级 `D0`。

### 5.2 两套枢轴不可混用

局部枢轴半径 `r`：

```text
pivot_high(i,r) if H_i > max(H_{i-r:i}) and H_i > max(H_{i+1:i+r+1})
pivot_low(i,r)  if L_i < min(L_{i-r:i}) and L_i < min(L_{i+1:i+r+1})
```

- `r=1`：用于近端支撑/阻力、结构目标与形态腿，最早在 `i+1` 发布，来源 `market_features.py` / `structure_levels.py`，`D0`。
- `r=2`：用于方向投票与较稳健的 HH/HL、LL/LH，最早在 `i+2` 发布，来源 `decision_nodes._find_swings`，`D0`。
- 连续出现同类枢轴时，折叠为该段更极端的一个，再形成高低交替的腿序列；此折叠是 `D1` 形式化，防止一个宽 K 线同时生成高低枢轴而制造假腿。

最近两个已确认高枢轴与低枢轴分别比较：

```text
HH = high_latest > high_previous
LH = high_latest < high_previous
HL = low_latest  > low_previous
LL = low_latest  < low_previous
```

`HH+HL` 为多头波段结构，`LL+LH` 为空头波段结构，否则 `mixed`。相等时不算 HH/LH/HL/LL。等级 `D0`。

### 5.3 支撑、阻力与结构失效候选

- 在最近 40 根中取 `r=1` 已确认 pivot low 且 `<C_t` 的价位，去重后由近到远最多 3 个支撑。
- 对 pivot high 且 `>C_t` 对称取得最多 3 个阻力。
- 没有局部枢轴时，上游会退化为所有低点/高点；阶段 1B 保留退化结果但标记 `fallback=true`，不得与确认枢轴等权。
- `invalidation_long_candidate` 是最近下方确认支撑；`invalidation_short_candidate` 是最近上方确认阻力。它们只是结构锚，不是最终 SL；缓冲和经纪商约束属于阶段 2/5。

等级 `D0-C`：上游的支撑/阻力算法可移植，但必须保留确认时间和 fallback 标记。

### 5.4 五信号方向投票

方向投票主窗口为最近 8 根，得分 `score = S1+...+S5`，每项为 `-1/0/+1`：

1. `S1 EMA slope`：`d=EMA20_t-EMA20_{t-k}`，`k=min(10,可用历史)`；`d>0.05*ATR` 得 `+1`，`d<-0.05*ATR` 得 `-1`。
2. `S2 close gravity`：8 根按新到旧赋权 `8..1`，分别计算近半与远半加权收盘均值；差值 `>0.10*ATR` 得 `+1`，`<-0.10*ATR` 得 `-1`。
3. `S3 swing`：8 根内 `r=2` 的最近两高两低形成 `HH+HL` 得 `+1`，`LL+LH` 得 `-1`。
4. `S4 trend bars`：多头/空头趋势棒数量之比达到 `1.5` 得对应 `+1/-1`；一方非零、另一方为零也得对应分。
5. `S5 overlap`：8 根平均相邻重叠 `<0.45` 时，按 `S1` 的方向再加 `+1/-1`；`>0.65` 或 EMA 斜率中性时为 0。

20 根中窗口用同样的近远半加权收盘重心做背景确认。若其方向与 8 根得分冲突且 `abs(score)<4`，把 `score` 向零减 1；`abs(score)>=4` 时近期新方向优先，不扣分。

```text
direction = bullish if score >= 3
direction = bearish if score <= -3
direction = neutral otherwise
direction_strength = abs(score)/5
```

等级 `D0`。上游注释曾把部分窗口写成 20，但实际代码主窗口为 8；本目录以固定提交的实际可执行参数为准。

### 5.5 Always-In

对窗口 `N`，最新 K 线权重为 `N`、最老为 1，只对 EMA 有效的 K 线计权：

```text
above_ratio = sum(weight where C>EMA) / sum(valid weights)
below_ratio = sum(weight where C<EMA) / sum(valid weights)
```

近端主判：`N=8`，同侧阈值 `0.65`，EMA 斜率回看 5 根并使用 `0.05*ATR` 死区。

背景回退：`N=20`，同侧阈值 `0.70`，EMA 斜率回看 10 根并使用相同死区。

```text
AIL core = above_ratio >= threshold and EMA slope > dead-zone
AIS core = below_ratio >= threshold and EMA slope < -dead-zone
```

近端 core 优先；近端无 core 时才允许使用背景 core，标记为 `weak`。结构强度附加条件为相应 `HH+HL/LL+LH` 且窗口最大收盘跨度 `<=1.5*ATR`；它只区分 strong/weak，不否定 core。等级 `D0`。

### 5.6 回撤、通道与区间状态

上游 `market_features._pullback_metrics` 把按 `seq` 升序后的尾元素误当成“最近枢轴”，会在 newest-first 数据中选到旧枢轴。阶段 1B 不移植这一方向缺陷，改用最近已确认的交替枢轴，等级 `D0-C`。

对多头完成腿 `low_origin -> high_impulse -> low_pullback`：

```text
pullback_ratio = (high_impulse-low_pullback) / (high_impulse-low_origin)
```

空头对称：

```text
pullback_ratio = (high_pullback-low_impulse) / (high_origin-low_impulse)
```

分母 `<=max(tick,eps)`、枢轴顺序不完整或回撤尚未确认时无效。比例不强制裁剪；`<0` 或 `>1` 分别说明无回撤或原腿已被穿越，不能静默归入通道。

通道候选使用 `r=2` 交替枢轴：

- 至少 3 组连续 `HH+HL` 或 `LL+LH` 才能成为已确认通道；只有 2 组为 `trending_tr` 候选。
- 最近完成回撤 `<0.30` 为 `tight_channel`；`[0.30,0.50]` 为 `normal_channel`；`(0.50,0.786]` 为 `broad_channel`。
- 对高枢轴和低枢轴分别按时间做最小二乘直线；`parallel_error = abs(slope_high-slope_low)/(abs(slope_high)+abs(slope_low)+eps)`。
- 提议阈值：`parallel_error<=0.35` 且两条线最大归一化残差均 `<=0.75*ATR` 才称“可画稳定通道线”；否则降级为 `trending_tr`。

前三项来自 playbook，直线稳定性阈值是为替代主观“可画线”的 `D1` 明确定义，尚无上游等价代码。

普通区间候选：

- 不满足 3 组单向波段序列；
- 上边界与下边界在容差 `tol_level=max(2*tick,0.15*ATR)` 内各有至少 2 个已确认测试；
- 方向投票为 neutral，或波段序列为 mixed；
- 中部/上下三分之一沿用 5.1。

极端区间不是普通区间的同义词。只有 EMA 10 根斜率 `<=0.05*ATR`、平均重叠 `>=0.70`、简化方向分绝对值 `<=1` 三项同时满足时，才输出 `extreme_tr_candidate=true`；最终是否禁用候选属于阶段 2 政策。前述阈值来自 `decision_nodes.py`，等级 `D0/D1` 组合。

### 5.7 Barbwire

最近 10 根：

```text
overlap_mean_10 = mean(overlap)
doji_inside_ratio_10 = count(doji or inside)/10
width_10 = max(H)-min(L)
avg_bar_range_10 = mean(H-L)

score = 0
+0.4 if overlap_mean_10 >= 0.65
+0.2 if doji_inside_ratio_10 >= 0.40
+0.2 if range_width_40/ATR14 <= 3.0
+0.2 if width_10/avg_bar_range_10 < 0.30
barbwire_candidate = score >= 0.60
```

有效项缺失时该项不加分，并在明细中标记缺失。该公式是 `D0`。playbook 的“区间宽度/平均波段高度 <25%”使用了未实现的“平均波段高度”，不能与 `width_10/avg_bar_range_10` 混为同一公式；阶段 1B 只实现上述可复现 score，25% 规则保留为未启用的 `D1` 备选。

## 6. 突破、回测与 H/L 计数

### 6.1 因果突破事件

在同一分段、最多 40 根的时间升序扫描中，令 `running_high/low` 只包含当前棒之前的数据：

- 向上突破：`C_i > running_high + tick`。
- 向下突破：`C_i < running_low - tick`。
- 向上突破失败：突破后最多 5 根内 `C_j < breakout_level-tick`。
- 向下突破失败：突破后最多 5 根内 `C_j > breakout_level+tick`。
- 向上回测成功：突破后最多 5 根内 `L_j <= level+0.15*ATR` 且 `C_j>level`。
- 向下回测成功：突破后最多 5 根内 `H_j >= level-0.15*ATR` 且 `C_j<level`。

同一价位只保留离当前最近的事件；输出事件必须记录突破棒、确认棒与价位。等级 `D0`。

`failed_failure/breakout_pullback` 在上游只有 playbook 语义，阶段 1B 形式化为：一次已确认失败突破之后 5 根内，收盘重新越过原突破位且随后 1–2 根满足同向 `follow=yes`。没有跟随只记 `rebreak_pending`。等级 `D1`。

### 6.2 H1/H2/L1/L2 状态机

上游 `_compute_hl_count` 只是“逐棒突破前棒极点”的累计计数，并不识别两腿回撤，也不会完整执行 playbook 的重置规则。因此可保留为诊断指标 `prior_extreme_break_count`，但不得直接暴露为 H1/H2/L1/L2，等级 `D0-C`。

规范状态机（`D1`）：

- 多头背景必须满足 `direction=bullish` 或 `AIL`。在一个已确认高枢轴之后，第一根满足 `L_i<L_{i-1}` 或 `C_i<C_{i-1}` 的棒启动回撤态；进入回撤态后，第一根满足 `H_i > H_{i-1}+tick` 的已收盘触发棒为 `H1`。
- 若 H1 后没有同向跟随，且形成新的已确认回撤低点，再次满足前棒高点突破时为 `H2`。
- 空头背景对称，以 `L_i < L_{i-1}-tick` 定义 `L1/L2`。
- 每次触发必须记录其所属回撤腿起止、触发棒和确认棒；不能仅返回数字。
- 新的强突破（收盘越过原趋势极点、棒长 `>=1.2*ATR`）并有跟随、Always-In 翻转、原趋势结构失效或数据段切换会清零计数。
- 第三次触发输出 `H3/L3` 与 `wedge_check_required=true`，不伪装成 H2/L2。

H1/L1 是第一次触发，不等价于“回撤只有一根”；H2/L2 必须证明第一次尝试与第二个回撤腿都真实存在。

## 7. 复合 PA 结构

这些结构在固定提交中主要由模型阅读 playbook 后判断。阶段 1B 实现的是本目录给出的确定性候选，不宣称与上游 AI 标签逐笔等价。

### 7.1 楔形与三推

使用已确认、时间有序的交替 `r=1` 枢轴。向上三推取四个锚点 `p0,p1,p2,p3`，其中 `p1..p3` 为三个递增高枢轴，推进量：

```text
a1 = p1-p0
a2 = p2-p1
a3 = p3-p2
```

向下对称使用绝对推进量。标准楔形候选必须同时满足：

- `a1,a2,a3 > max(tick,eps)`；
- `a2 <= 1.05*a1`、`a3 <= 1.05*a2` 且 `a3/a1 <= 0.80`；
- 对应反向枢轴也朝同一大方向移动；
- 两侧回归线在推进方向收敛：向上楔形须 `slope_low>slope_high>0`，向下楔形须 `slope_high<slope_low<0`，且末端线间距小于起点线间距；每条线最大残差 `<=0.75*ATR`；
- 三推跨度 10–40 根；3–4 根仅标 `micro_wedge`，不得单独升级为标准楔形。

与主趋势反向、跨度 `<20` 的候选为 `wedge_pullback`；与主趋势同向且跨度 `>=20` 为 `wedge_reversal_candidate`。第四推扩大到 `>1.10*a3` 且有跟随，或任一边界被反向收盘突破后立即收回，会使当前楔形候选失效/降级。等级 `D1`。

### 7.2 Always-In、趋势/通道与区间环境

环境标签由第 5 节组合，不再让模型自由发明：

- `spike_candidate`：至少 2 根同向趋势棒；相邻实体重叠 `<0.30`；连续收盘创新同向极值。3–5 根且重叠 `<0.20` 为 `standard_spike`。
- 6 根以上只产生 `climax_warning`；任意强序列后出现尾线 `>0.50*body`、实体 `<0.30*近期平均实体` 或反向趋势棒，产生 `climax_triggered`。
- 多头 `micro_channel`：取 2–10 根窗口，至少 `n-2` 个相邻转换同时满足 `H_i>=H_{i-1}` 与 `L_i>=L_{i-1}`；空头对称使用 `<=`。其余最多 2 个转换只能由 inside/doji 小暂停造成，且反向位移不得超过 `0.25*ATR`；同时必须未达到 spike 标准。
- `tight/normal/broad_channel`、`trending_tr`、`trading_range` 按 5.6。

`spike` 的“实体重叠”按实体区间的交集/并集计算，不使用整根高低区间 overlap；这是 playbook 的 `D1` 形式化。climax 只输出环境风险，不直接生成反向信号。

### 7.3 MTR

`mtr_candidate` 必须四组件齐全：

1. 原趋势：方向投票非 neutral，并有 `HH+HL` 或 `LL+LH`。
2. 趋势线/通道线突破：收盘越过经确认的主趋势线至少 `tick`，不是单根影线。
3. 原趋势恢复失败：突破后没有同向新极值与跟随，或恢复尝试被 1–2 根反向收盘否定。
4. 前极点测试失败：价格在 `tol_level` 内二次测试原极点，但收盘未确认新极值，并形成较低高点/较高低点。

输出每个组件的布尔值、证据 K 线和确认时间；缺一只能是 `reversal_attempt`。新收盘价重新突破原趋势极点并有跟随时，候选失效。等级 `D1`。

PA_Agent 的提示词政策规定 MTR 只诊断、禁止逆势三价；这是 `P`，不是公式。AlphaMaster 已批准的“反转策略家族”是否使用该候选，由阶段 2 的候选政策决定，阶段 1B 只输出诊断证据，不输出订单。

### 7.4 双顶、双底与微结构

- `micro_double` 保持 4.4 的 `D0` 公式。
- 完整双顶/底使用两个同类 `r=1` 已确认枢轴，容差 `tol_double=max(2*tick,0.10*ATR)`。
- 两枢轴之间必须存在反向枢轴，颈线深度至少 `0.50*ATR`；否则只记 `near_equal_extremes`。
- 第二次测试后必须有反向趋势棒或失败突破事件，才成为 `double_top_candidate/double_bottom_candidate`。
- 收盘越过第二极点 `tol_double` 且有跟随时失效。

等级 `D1`。`0.10*ATR` 与 `0.50*ATR` 是为“相近极点”和“清晰回撤”提供可复现边界的 AlphaMaster 形式化，不是上游代码已有阈值。

### 7.5 Final Flag

`final_flag_candidate` 必须满足：

- 前置方向投票绝对分 `>=3`，且近 20 根曾有同向 Always-In 或标准 spike；
- 随后出现至少 10 根水平/近水平整理；整理段 `abs(EMA slope 10)/ATR <=0.10`；
- 整理段平均 overlap `>=0.50`，doji/inside 比例 `>=0.40`；
- 整理位于前趋势极点或有效 MM 目标 `<=0.50*ATR` 的邻域。

顺原趋势突破后 1–2 根无跟随并收回整理区，输出 `failed_final_flag=true`。整理向原趋势方向重新扩张且连续 2 根有跟随时，当前 final flag 失败候选失效。等级 `D1`。

PA_Agent 禁止追 FF 和禁止逆势 FFES 是执行政策 `P`；阶段 1B 只产出候选和失败证据。

### 7.6 测量目标

区间高度：

```text
height = range_high-range_low
range_mm_up = range_high+height
range_mm_down = range_low-height
```

完成腿投影：

```text
leg_height = abs(leg_end-leg_start)
continuation_mm = pullback_end + sign(leg)*leg_height
```

通道投影使用同一时点两条平行线的垂直价差 `channel_width`：突破点沿突破方向加/减该宽度。楔形投影使用楔形起点到极点的垂直高度，从确认突破点沿突破方向投射。

上游 `market_features._measured_move_candidates` 的 `leg_up/leg_down` 未验证高低枢轴的时间顺序，且从当前 close 同时向两边投影。阶段 1B 必须先证明腿已按时间完成，再从对应回撤终点投影；无完整腿时只允许区间 MM。等级 `D0-C/D1`。

测量目标只是价格锚；最终 TP1/TP2、最低 R 和经纪商距离属于阶段 2/5。

## 8. 四类 PA 策略的公式边界

阶段 1B 输出“结构候选”，不生成 Market/Limit/Stop 订单，不决定手数、SL/TP 缓冲，也不调用 Codex。

### 8.1 趋势延续候选

硬门：

- `direction` 与 `Always-In` 不冲突；
- 环境为 spike 回撤、micro/tight/normal/broad channel 中之一；
- 有 H1/H2/L1/L2、wedge pullback、EMA 回撤或反向假突破失败中的至少一项；
- 非 barbwire，中部区间不激活；
- 有已确认结构失效锚和至少一个顺向结构目标。

输出方向、setup 类型、触发证据、失效锚和目标锚。`climax_triggered` 时仍可保留候选证据，但标 `chase_forbidden=true`；是否等待回撤由阶段 2 决定。

### 8.2 突破候选

硬门：

- 有 6.1 的收盘突破；
- 突破方向与当前方向证据不冲突；
- 满足 `follow=yes`，并有回测成功或 `failed_failure`；
- 普通区间第一根突破只输出 `pending`，不输出可执行候选；
- 有突破位、回测极点和 MM 锚。

### 8.3 反转候选

硬门：

- `mtr_candidate` 四组件齐全；
- 至少再有 wedge reversal、double top/bottom 或 failed final flag 之一；
- 反向信号棒已收盘且有跟随；
- 输出必须带 `source_policy=diagnostic_only`，提醒上游 PA_Agent 原政策不允许逆势订单。

阶段 1B 不把此候选升级为交易许可。阶段 2 若按已批准的 AlphaMaster 反转策略家族使用它，必须单独记录这是 AlphaMaster 新政策而非 PA_Agent 原执行语义。

### 8.4 区间交易候选

硬门：

- 普通区间边界各至少 2 次确认；
- 当前价格在上/下三分之一，不在中部；
- 非 barbwire/extreme_tr；
- 有与候选方向一致的第二次边界测试、H2/L2 或失败突破；
- 有边界失效锚与对边/近端结构目标。

方向 neutral 时公式层可输出两侧的只读证据，但不得输出“已许可方向”；阶段 2 的多周期方向融合只能选一侧或拒绝，不能同时下双向候选。

## 9. 明确不照搬的上游行为

| 上游行为 | 问题 | 本目录裁决 |
|---|---|---|
| newest-first `pullback_metrics` 使用排序尾枢轴 | 实际可能取最旧枢轴 | 使用最近已确认、时间有序的交替枢轴 |
| `_compute_hl_count` 直接称 H1/H2/L1/L2 | 只累计前棒极点突破，没有两腿语义 | 改名诊断计数；真正 H/L 使用 6.2 状态机 |
| `bull_gap/bear_gap` 同时承载几何侧别与背景含义 | 名称歧义 | 规范为 `ema_gap_above/below` |
| `r=1` 与 `r=2` 枢轴未在接口层区分 | 相同 HH/HL 名称可能得出不同结果 | 结果必须记录 `pivot_radius` |
| leg MM 同时从当前价向两边投影 | 未验证腿方向与完成顺序 | 只对已完成时间有序腿投影 |
| AI 自由判断 wedge/MTR/FF | 不可复现且无法做因果回测 | 使用第 7 节 D1 公式，并标明不保证 AI 标签等价 |
| MTR、FF、双顶底一律禁止逆势订单 | 这是 PA_Agent 的执行政策，不是形态公式 | 阶段 1B 保留 `diagnostic_only` 来源标记；阶段 2 单独决策 |
| 输出仓位、分批、移止损的 Brooks 叙述 | 与 PA_Agent 自身硬禁令及当前阶段不一致 | 完全不进入公式引擎 |

## 10. 阶段 1B 接口与 golden fixtures

### 10.1 最小接口合同

阶段 1B 应提供三个纯计算层，名称可在实施方案中调整，但职责不得混合：

1. `BarFormulaEngine`：第 4 节单棒/多棒几何、EMA、ATR。
2. `StructureFormulaEngine`：第 5–7 节枢轴、方向、Always-In、环境和复合结构。
3. `PAStrategyDetector`：第 8 节四类只读候选。

所有 API 接受时间升序、同一品种/周期、已收盘且同段的 K 线；输出不可变结果。公式层不得读取网页配置、账户状态、Codex、MT5 订单状态或未来 K 线。

### 10.2 必须交付的 golden fixtures

阶段 1B 至少包含以下人工可核对样例：

1. 单棒几何：趋势棒、doji、inside、outside、零区间棒。
2. 多棒组合：`ii`、`iii`、`ioi`，并验证时间升序与上游 newest-first 映射一致。
3. ATR14/EMA20：整段与增量逐点一致，预热位置精确。
4. 微双底/顶、EMA gap run 与 20GB 边界 `19/20/21`。
5. `r=1/r=2` 枢轴确认延迟；修改确认点之后的数据不得改变确认点之前输出。
6. HH+HL、LL+LH、mixed；支撑/阻力 fallback 必须显式标记。
7. 五投票方向分数 `2/3/4` 与 `-2/-3/-4` 阈值。
8. Always-In 近端 65%、背景 70%、近端/背景冲突与 1.5 ATR 强弱边界。
9. tight/normal/broad 的 `0.30/0.50/0.786` 精确边界和两组序列降级 trending_tr。
10. Barbwire score 每个分量以及 `0.59/0.60` 候选边界。
11. 突破、5 根内失败、0.15 ATR 回测、失败的失败。
12. H1→失败→第二腿→H2 与重置；第三次触发转 wedge check。
13. wedge pullback/reversal、完整/缺组件 MTR、双顶底、failed final flag。
14. 区间/腿/通道/楔形 MM，拒绝时间顺序错误的腿。
15. 四类策略候选各一组阳性和硬门阴性样例。
16. 全套前缀不变测试：在截断点之后任意修改 OHLC，不改变截断点及之前任何输出。

### 10.3 阶段 1B 完成判据

- 所有 D0 公式均有与固定提交相符的 golden 证据。
- 所有 D0-C 修正均有“上游缺陷复现 + 修正结果”证据。
- 所有 D1 公式均有阈值内、等于阈值、阈值外三类证据。
- 每个复合候选可追溯到确切 K 线、确认时间和结构价位。
- 没有形成中 K 线、未来数据、跨缺口窗口或成交量依赖。
- 没有订单发送、账户写入、Codex 调用、网页改动或仓位管理代码。
- AGPL 来源说明已进入相应生产模块和项目归属文档。

## 11. 本阶段冻结结论

可直接移植的 `D0` 范围：K 线几何、ATR14、EMA20、inside/outside/ii/iii/ioi、相邻重叠、基础趋势棒、跟随、区间位置、Barbwire score、因果突破/回测、方向投票、Always-In、基础枢轴和支撑阻力。

需按本目录修正的 `D0-C` 范围：时间方向、回撤枢轴、H/L 计数命名、EMA gap 命名、枢轴半径、完成腿 MM。

需新形式化的 `D1` 范围：通道稳定性、真正 H1/H2/L1/L2 状态机、失败的失败、wedge、MTR、完整双顶底、Final Flag 与四类 PA 策略候选。

阶段 1A 不声称这些候选能提升策略准确率；它只把原本由提示词/模型主观判断的内容变成可复现、可回测、无未来数据的合同。准确率与权重必须在阶段 2 的组合消融和阶段 6 的保留周回测中验证。
