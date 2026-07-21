# Index 组训练参数调优 — 2026-07-06 10:50

## 目标
第一轮训练（09:46启动）在 step 220 熵坍塌（H: 3.8→0.6），停掉后调整参数重训

## 第一轮问题诊断
- 熵从 3.8 快速坍塌到 0.3（100步内）
- ENTROPY_FLOOR_LAMBDA=2.0 力度不够
- ENTROPY_COEFF_POWER=1.3 导致低熵时系数不够激进
- 重启后从 best_snapshot 恢复，吸引子效应导致快速再次坍塌
- top1=1.0, effV=1.0（完全锁死单一公式）

## 参数调整
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| ENTROPY_FLOOR_THRESH | 0.5 | 1.0 | 更早介入 |
| ENTROPY_FLOOR_LAMBDA | 2.0 | 5.0 | 加大惩罚力度 5x |
| ENTROPY_COEFF_POWER | 1.3 | 1.0 | 低熵时系数更激进 |
| ENTROPY_COLLAPSE_STEPS | 40 | 20 | 更快检测坍塌 |
| 深度坍塌强制 full reset | 无 | H<0.3 时强制 | 避免吸引子效应 |

## 效果预期
- H=0 时 floor_loss: 旧=1.0, 新=5.0 (5x)
- H=0.5 时 floor_loss: 旧=1.0, 新=2.5 (2.5x)
- H=1.0 时不触发（新阈值）
- ent_coeff at H=0.1: 旧=0.88, 新=0.91 (微增)
- 深度坍塌 (H<0.3) 直接 full reset 不从 best 恢复

## 执行
- 清除旧 checkpoint (11个 ckpt_index_step_*.pt)
- 启动新训练 session: swift-fjord, PID 33260
- Step 1-3: H=3.82-3.85（健康），c=0.207（比旧 0.130 更激进）
