# CORE-07 可复现构建与发布门禁

## 支持范围与安装

核心训练、Parquet 数据、回测、features/ops/VM/execution 支持 Python
3.10、3.11 和 3.12，并使用 `constraints-core.txt` 中的直接依赖锁：

```bash
python -m pip install -r requirements.txt -c constraints-core.txt
```

该命令不会安装 Windows 或 MT5 依赖。只有需要 MT5 adapter 的 Windows
环境才使用：

```bash
python -m pip install -r requirements-mt5.txt
```

开发与 CI 工具使用：

```bash
python -m pip install -r requirements-dev.txt -c constraints-core.txt
```

## Required CI

`.github/workflows/core-required.yml` 定义五个 required job：

- `lint-type`：关键错误 lint 与核心边界类型检查；
- `core-unit`：Ubuntu 上 Python 3.10/3.11/3.12 的核心、性质和 coverage 门禁；
- `windows-core`：Windows 文件、进程、stub 与导入回归；
- `deterministic`：Ubuntu/Windows 的 reference trace 和精确恢复等价；
- `integration`：短训练、策略发布、replay 与独立 OOS 全链路。

所有 job 都有硬超时。pytest 默认把未列入白名单的新 warning 当作错误；失败时
输出 seed、formula、dataset identity 与 trace digest（不可取得的字段明确显示
`unavailable`）。

## 发布与重训

发布或正式重训入口必须先执行：

```bash
python scripts/release_gate.py
```

只有聚合门禁设置 `CORE_REQUIRED_GATES=passed` 时命令才成功；缺失、失败或其他
值都 fail closed。该信号只能由 required CI 聚合步骤提供，不应由训练代码静默
补值。
