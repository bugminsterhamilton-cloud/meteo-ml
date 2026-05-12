from __future__ import annotations

import argparse

import uvicorn

from app.api.app import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the Meteo ML forecast service.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        "app.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
