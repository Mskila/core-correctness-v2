# 训练重启 — 2026-07-06 09:46

## 目标
应用 anti-beta 约束 + 熵下限修复 + EMA Reward Baseline 后，从头重新训练三组策略

## 前置修复（已完成）
1. **熵下限惩罚**: ENTROPY_FLOOR_THRESH=0.5, ENTROPY_FLOOR_LAMBDA=2.0
2. **重启多样性**: FULL_RESET_EVERY=3（每第3次 restart 全参数 reset）
3. **EMA Reward Baseline**: REWARD_EMA_DECAY=0.95（替代 batch mean 计算 advantage）
4. **Anti-beta 约束**: 感染链追踪，禁止恒正算子连续出现/末尾出现
5. **Beta 中性惩罚**: >85% 同向减半 reward
6. **前后一致性约束**: 前后半段年化同号奖励

## 执行
- 清除旧 checkpoint（移至 `checkpoints_archive_20250706/`）
- 启动 index 组训练: `python main.py --offline --group index`
- PID: 13160, session: wild-crest
- 数据: 5 品种 (US30.cash, US100.cash, US500.cash, US2000.cash, JP225.cash), T=32076 bars (5.15年)
- 训练目标: 5000 steps

## 早期状态 (step 3)
- H=3.808（健康，修复前会快速坍塌）
- Best=0.162 (step 0)
- utok=90/90, effV=25（搜索空间未过度限制）
- Reward=-0.483（正常，初期需要探索）

## 训练顺序
1. **index**（当前）— 最需要重训，旧策略无效
2. **metals_comm** — 数据短但重训看新约束效果
3. **forex** — 已有两版可用策略，进一步优化

## 监控
- 设有每 30 分钟 cron 自动检查进度
- 训练完成后会自动保存策略到 `strategies/` 并打印结果
