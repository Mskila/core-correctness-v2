"""
model_core/config.py — 模型层配置

仅保留模型训练所需的参数。
品种、数据、风控等全局配置统一由根目录 config.py 的 Config 类管理。
"""
import math

import torch
from .vocab import FORMULA_VOCAB


class ModelConfig:
    RANDOM_SEED = 42
    USE_LORD_REGULARIZATION: bool = True
    LORD_DECAY_RATE: float = 1.0e-3
    LORD_NUM_ITERATIONS: int = 5
    # ── 训练设备 ─────────────────────────────────────────────────────────
    # 注意：本任务 CPU 训练速度反而比 GPU 快（实测约 2.3 倍），故强制用 CPU。
    # 原因：
    #   1. 张量太小——forex 组仅 (2 品种 × 3508 × 20 特征)，单个算子的
    #      计算量小于 CUDA kernel 启动开销（数十微秒），GPU 算得快但启动慢。
    #   2. 训练循环是 Python 串行调度：每 step 逐条跑 128 条公式 × 8 个
    #      VM 步 × 4 个 walk-forward 折，GPU 被切成上万个碎片段，吃不满。
    #   3. host↔device 拷贝 + kernel 启动延迟主导总耗时，而非张量计算本身。
    #   4. 实测 GPU 利用率 ~51%，正是 GPU 一半时间在干等 Python 喂下一个
    #      kernel 的证据（不是“还能压榨”，而是“调度瓶颈”）。
    # 基准测试（forex 组, 50 步, RTX 4060, 2026-07-03）：
    #   cuda: 4.48 s/步  Best=4.875
    #   cpu : 1.91 s/步  Best=5.103
    #   加速比 = 0.43x（GPU 反而慢 2.3 倍）
    # 若后续改为批量并行公式评估（一次喂大批张量进 GPU），再切回 cuda。
    DEVICE = torch.device("cpu")

    # ── 训练参数（大搜索空间适配版，2026-07-04 重构）─────────────────────
    # 背景：特征库扩展到 65、算子库扩展到 66（vocab=131），8-token 搜索空间
    #   从旧版 ~7亿 暴增到 ~8.67×10^16（1.2 亿倍）。旧的采样预算（128×3000）
    #   覆盖率趋近于零，导致熵坍塌 Early Stop、公式退化。
    # 对策（训练时间不敏感场景）：
    #   1. 特征剪枝（active_features.json）把 vocab 降到 ~90，空间缩小约 20 倍
    #   2. 放大采样预算：BATCH_SIZE 128→256，TRAIN_STEPS 3000→8000
    #   3. 更大精英池（60）保留更多历史最优
    BATCH_SIZE      = 192   # 每步采样公式数（原 128，1.5x 提升覆盖率）
    TRAIN_STEPS     = 9000  # 每组训练步数（55次重启需要更多步数）
    MAX_FORMULA_LEN = 8     # 公式长度上限：保持 8（10 会导致 CPU 训练慢 3 倍）

    # ── 特征维度（由 vocab.py 自动派生，无需手动修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 为主，IC 做门控 ──────────────────────────────────
    # IC_NEG_MULT 0.30→0.50：0.30 对反向因子惩罚过重，可能误杀非线性高收益因子。
    # 收益优先模式下，只要年化收益是正的，适当负 IC 可以接受。
    REWARD_ALPHA:      float = 1.0
    IC_GATE_THRESH:    float = 0.01
    IC_GATE_MULT:      float = 1.15
    IC_NEG_MULT:       float = 0.75   # 收益优先：不过度误杀反向/非线性高收益因子
    IC_GATE_SCALE_FLOOR: float = 0.0
    OOS_GATE_SCALE: float = 0.5
    TARGET_TRADES_PER_DAY: float = 2.0

    # ── FTMO 专属奖励模式 ─────────────────────────────────────────────
    # "standard": 收益+风险平衡（默认，原权重）
    # "ftmo":     FTMO 考试盘专属——年化收益权重 0.60→0.75，Calmar 0.05→0.10
    #             （控制 MDD 贴近 10% Max Loss 上限），其余指标权重下调。
    #             目标：在 10% Max Loss 约束下最大化年化收益，快速达标。
    # "forex":    外汇均值回归专属（2026-07-08）——
    #             降年化收益权重(0.80→0.25)、提IC权重(0.03→0.25)、
    #             新增反转奖励(0.20，奖励低/负因子自相关)和多空对称检查(0.15)。
    #             原因：外汇H1以震荡为主，趋势算子效果差，需引导模型偏好
    #             均值回归信号而非追涨杀跌。
    REWARD_MODE:       str = "ftmo"

    # ── 熵保护（大空间加强版）──────────────────────────────────────────
    # ENTROPY_COEFF_MAX 0.5→1.0：加倍探索压力，对抗大 vocab 的过早收敛。
    # ENTROPY_COLLAPSE_THRESH 改为相对阈值 0.15×ln(vocab)：大 vocab 最大熵更高
    #   （ln(131)≈4.87 vs ln(54)≈3.99），绝对阈值 0.5 不再合理。
    # ENTROPY_COLLAPSE_STEPS 15→40：给模型更长的自我恢复窗口，不急于重启。
    ENTROPY_COEFF_MAX:   float = 1.0
    ENTROPY_COEFF_POWER: float = 1.0  # 降低幂次，让低熵时系数更激进（原1.3）
    ENTROPY_COLLAPSE_THRESH: float = 0.15 * math.log(FORMULA_VOCAB.size)
    ENTROPY_COLLAPSE_STEPS:  int   = 20  # 更快检测坍塌并重启

    # ── 熵下限惩罚（Fix 1: H→0 时熵项归零问题）──────────────────────────
    # 当 H < ENTROPY_FLOOR_THRESH 时，加入固定惩罚 λ×(thresh-H)。
    # 这确保即使 mean_ent→0，loss 中仍有非零探索压力。
    ENTROPY_FLOOR:        bool  = True
    ENTROPY_FLOOR_THRESH: float = 1.0   # 熵低于此值时触发固定惩罚（提高介入时机）
    ENTROPY_FLOOR_LAMBDA: float = 5.0   # 惩罚强度系数（加大力度对抗坍塌）

    # ── Elite Replay ──────────────────────────────────────────────────
    ELITE_REPLAY_FRAC:  float = 0.25
    ELITE_POOL_SIZE:    int   = 60    # 30→60：大空间需要更大的精英记忆
    ELITE_REWARD_SCALE: float = 1.2

    # ── 坍塌重启（大空间加强版）─────────────────────────────────────────
    # MAX_RESTARTS 8→25→55、RESTART_NOISE 0.05→0.1→0.25：时间不敏感，多给机会+更强扰动。
    # 配合 engine.py：超过 MAX_RESTARTS 后不再 Early Stop，改为强扰动继续训练。
    # 2026-07-09: US100 训练 24/25 重启仍有突破，扩到 55 次。
    MAX_RESTARTS:   int   = 55
    RESTART_NOISE:  float = 0.25

    # ── 自适应噪声：Best 停滞时自动增大扰动 ─────────────────────────────
    # stagnation_window: 判断停滞的步数窗口
    # noise_min / noise_max: 噪声下界和上界
    # noise_boost: 停滞时噪声提升倍率
    ADAPTIVE_NOISE:      bool  = True
    STAGNATION_WINDOW:   int   = 500
    NOISE_MIN:           float = 0.15
    NOISE_MAX:           float = 0.60
    NOISE_BOOST_FACTOR:  float = 2.0   # noise += 0.2 * (stagnation / window)

    # ── 重启多样性（Fix 2: best_snapshot 吸引子效应）─────────────────────
    # 每 FULL_RESET_EVERY 次重启中，做 1 次完全随机初始化而非从 best_snapshot 恢复。
    FULL_RESET_EVERY:    int   = 3     # 每 3 次重启中第 3 次做 full reset

    # ── Reward baseline（Fix 3: 全负 batch 相对优选问题）──────────────────
    # 用 EMA baseline 替代 batch mean 计算 advantage，避免全负 batch 的问题。
    REWARD_EMA_BASELINE:     bool  = True
    REWARD_EMA_DECAY:        float = 0.95   # EMA 衰减系数
    REWARD_EMA_WARMUP:       int   = 10     # 前 N 步用 batch mean（EMA 未稳定）

    # ── 重启时部分重置参数：保留底层，扰动顶层 ───────────────────────────
    PARTIAL_RESET:       bool  = True
    PARTIAL_RESET_LAYERS: tuple = ("ln_f", "mtp_head", "head_critic", "blocks", "token_emb")

    # ── Elite Replay 衰减：旧 elite 采样权重随时间衰减 ──────────────────
    ELITE_DECAY:         bool  = True
    ELITE_DECAY_HALF_LIFE: int = 300   # 每 300 步旧 elite 权重减半

    # ── 多起点并行（Island）──────────────────────────────────────────────
    # 注意：Island 模式在 CPU 训练下会让总时间变成 N 倍（islands 串行），
    # 对于 index 这类大数组（T=32076）会变得极慢。当前默认关闭，保留配置开关。
    N_ISLANDS:              int   = 1
    MIGRATION_INTERVAL:     int   = 500
    MIGRATION_TOP_K:        int   = 5
    # island 默认关闭，避免用户误开导致速度爆炸

    # ── 因子去相关参数 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 25
    CORR_THRESHOLD:   float = 0.85
    CORR_PENALTY:     float = 0.8

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    WF_GAP: int = 20
    WF_N_BLOCKS: int = 5
    WF_MIN_FOLD_BARS: int = 200

    # ── 公式结构约束（2026-07-05 新增）──────────────────────────────────
    # 背景：index 组因子因 TS_RANK 连续使用退化为 beta 因子（91.8% 做多），
    # 前半段市场跌亏钱、后半段市场涨赚钱，不是 alpha 而是 beta。
    # 对策：在采样阶段禁止恒正算子链，在评分阶段添加 beta 中性 + 前后一致性奖惩。
    ENABLE_FORMULA_STRUCTURE_CONSTRAINT: bool = True   # 总开关
    BETA_NEUTRAL_PENALTY:     bool  = True             # 多空比例失衡惩罚
    HALF_CONSISTENCY_BONUS:   bool  = True             # 前后一致性奖惩
    BETA_NEUTRAL_THRESH:      float = 0.85             # 超过此比例同方向触发重罚
    BETA_NEUTRAL_LIGHT_THRESH: float = 0.70            # 轻度失衡阈值

    @classmethod
    def training_config_snapshot(
        cls,
        random_seed: int,
        *,
        timeframe: str = "H1",
    ) -> dict[str, object]:
        """Export the sole authoritative, identity-hashed training snapshot."""
        from config import Config
        from model_core.reward import target_bars_per_trade
        from model_core.semantics import LABEL_LOOKAHEAD_BARS
        from model_core.walk_forward import formula_warmup_bars

        return {
            "model": {
                "input_dim": cls.INPUT_DIM,
                "hidden_dim": 96,
                "num_layers": 3,
            },
            "batch_size": cls.BATCH_SIZE,
            "train_steps": cls.TRAIN_STEPS,
            "max_formula_len": cls.MAX_FORMULA_LEN,
            "reward": {
                "ic": 1.0,
                "return": 1.0,
                "alpha": cls.REWARD_ALPHA,
                "mode": cls.REWARD_MODE,
                "ic_gate_thresh": cls.IC_GATE_THRESH,
                "ic_gate_mult": cls.IC_GATE_MULT,
                "ic_neg_mult": cls.IC_NEG_MULT,
                "ic_gate_scale_floor": cls.IC_GATE_SCALE_FLOOR,
                "oos_gate_scale": cls.OOS_GATE_SCALE,
                "ema_baseline": cls.REWARD_EMA_BASELINE,
                "ema_decay": cls.REWARD_EMA_DECAY,
                "ema_warmup": cls.REWARD_EMA_WARMUP,
                "factor_top_k": cls.FACTOR_TOP_K,
                "corr_threshold": cls.CORR_THRESHOLD,
                "corr_penalty": cls.CORR_PENALTY,
                "beta_neutral_penalty": cls.BETA_NEUTRAL_PENALTY,
                "half_consistency_bonus": cls.HALF_CONSISTENCY_BONUS,
                "beta_neutral_thresh": cls.BETA_NEUTRAL_THRESH,
                "beta_neutral_light_thresh": cls.BETA_NEUTRAL_LIGHT_THRESH,
            },
            "timeframe_reward": {
                "timeframe": timeframe,
                "target_trades_per_day": cls.TARGET_TRADES_PER_DAY,
                "target_bars_per_trade": target_bars_per_trade(
                    timeframe,
                    cls.TARGET_TRADES_PER_DAY,
                ),
            },
            "entropy": {
                "coeff_max": cls.ENTROPY_COEFF_MAX,
                "coeff_power": cls.ENTROPY_COEFF_POWER,
                "collapse_thresh": cls.ENTROPY_COLLAPSE_THRESH,
                "collapse_steps": cls.ENTROPY_COLLAPSE_STEPS,
                "floor_enabled": cls.ENTROPY_FLOOR,
                "floor_thresh": cls.ENTROPY_FLOOR_THRESH,
                "floor_lambda": cls.ENTROPY_FLOOR_LAMBDA,
            },
            "elite": {
                "size": cls.ELITE_POOL_SIZE,
                "replay_frac": cls.ELITE_REPLAY_FRAC,
                "reward_scale": cls.ELITE_REWARD_SCALE,
                "decay": cls.ELITE_DECAY,
                "decay_half_life": cls.ELITE_DECAY_HALF_LIFE,
            },
            "restart": {
                "max_restarts": cls.MAX_RESTARTS,
                "restart_noise": cls.RESTART_NOISE,
                "stagnation_window": cls.STAGNATION_WINDOW,
                "full_reset_every": cls.FULL_RESET_EVERY,
                "partial_reset": cls.PARTIAL_RESET,
                "partial_reset_layers": list(cls.PARTIAL_RESET_LAYERS),
            },
            "noise": {
                "initial": cls.RESTART_NOISE,
                "boost": cls.NOISE_BOOST_FACTOR,
                "adaptive": cls.ADAPTIVE_NOISE,
                "min": cls.NOISE_MIN,
                "max": cls.NOISE_MAX,
                "boost_factor": cls.NOISE_BOOST_FACTOR,
            },
            "lord": {
                "use_lord_regularization": cls.USE_LORD_REGULARIZATION,
                "lord_decay_rate": cls.LORD_DECAY_RATE,
                "lord_num_iterations": cls.LORD_NUM_ITERATIONS,
            },
            "walk_forward": {
                "blocks": cls.WF_N_BLOCKS,
                "gap": cls.WF_GAP,
                "min_fold_bars": cls.WF_MIN_FOLD_BARS,
                "min_trade_events": max(5, cls.WF_MIN_FOLD_BARS // 100),
                "required_metrics": ["ic", "net_return", "sortino"],
                "warmup_bars": formula_warmup_bars(cls.MAX_FORMULA_LEN),
                "label_lookahead": LABEL_LOOKAHEAD_BARS,
            },
            "cost_rate": Config.COST_RATE,
            "neutral_band": Config.MIN_TRADE_EXPOSURE,
            "random_seed": random_seed,
        }
