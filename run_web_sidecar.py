"""Start a read-only Web view beside an already running AlphaMaster service."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaMaster read-only Web sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--origin", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    origin = urlsplit(args.origin)
    if origin.scheme not in {"http", "https"} or not origin.hostname:
        parser.error("--origin must be an absolute HTTP(S) URL")
    if origin.hostname in {args.host, "127.0.0.1", "localhost"}:
        origin_port = origin.port or (443 if origin.scheme == "https" else 80)
        if origin_port == args.port:
            parser.error("sidecar port must differ from the origin service port")

    try:
        import uvicorn
        import web.sidecar_app as sidecar_app
    except ImportError:
        print("请先安装依赖: pip install fastapi uvicorn[standard]")
        sys.exit(1)

    sidecar_app.ORIGIN_URL = args.origin.rstrip("/")
    app = sidecar_app.create_app()
    print("\n  AlphaMaster 训练曲线只读旁路")
    print(f"  → http://{args.host}:{args.port}")
    print(f"  镜像来源: {sidecar_app.ORIGIN_URL}")
    print("  旁路不提供训练、停止、导入、回测或设置修改操作\n")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
