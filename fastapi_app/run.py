from __future__ import annotations

import asyncio
import os
from pathlib import Path

import uvicorn


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _require_file(path: str, env_name: str) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{env_name} points to missing file: {path}")
    return str(resolved)


def _build_server(
    *,
    app: str,
    host: str,
    port: int,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
) -> uvicorn.Server:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "*"),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
    return uvicorn.Server(config)


async def _serve_all(servers: list[uvicorn.Server]) -> None:
    await asyncio.gather(*(server.serve() for server in servers))


def main() -> None:
    host = os.getenv("KGO_HOST", "0.0.0.0")
    certfile = os.getenv("KGO_SSL_CERTFILE")
    keyfile = os.getenv("KGO_SSL_KEYFILE")

    if bool(certfile) != bool(keyfile):
        raise RuntimeError("KGO_SSL_CERTFILE and KGO_SSL_KEYFILE must be set together.")

    if certfile and keyfile:
        https_port = _env_int("KGO_HTTPS_PORT", _env_int("KGO_PORT", 8443))
        http_port = _env_int("KGO_HTTP_PORT", 8000)

        os.environ.setdefault("KGO_FORCE_HTTPS", "1")
        os.environ.setdefault("KGO_HTTPS_PORT", str(https_port))

        servers = [
            _build_server(
                app="fastapi_app.main:app",
                host=host,
                port=https_port,
                ssl_certfile=_require_file(certfile, "KGO_SSL_CERTFILE"),
                ssl_keyfile=_require_file(keyfile, "KGO_SSL_KEYFILE"),
            )
        ]

        if _env_flag("KGO_HTTP_REDIRECT", True) and http_port != https_port:
            servers.append(_build_server(app="fastapi_app.main:app", host=host, port=http_port))

        asyncio.run(_serve_all(servers))
        return

    port = _env_int("KGO_PORT", 8000)
    server = _build_server(app="fastapi_app.main:app", host=host, port=port)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
