from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from starlette.datastructures import URL
from starlette.responses import RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class RedirectToHTTPSMiddleware:
    def __init__(self, app: ASGIApp, https_port: int | str | None = None) -> None:
        self.app = app
        self.https_port = str(https_port) if https_port else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("scheme") != "http":
            await self.app(scope, receive, send)
            return

        url = URL(scope=scope)
        redirect_url = str(url.replace(scheme="https"))

        if self.https_port:
            redirect_url = self._replace_netloc(redirect_url, scope)

        response = RedirectResponse(redirect_url, status_code=308)
        await response(scope, receive, send)

    def _replace_netloc(self, url: str, scope: Scope) -> str:
        parts = urlsplit(url)
        host_header = self._host_header(scope)
        host_parts = urlsplit(f"//{host_header}")
        hostname = host_parts.hostname or parts.hostname or "localhost"

        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"

        port = "" if self.https_port == "443" else f":{self.https_port}"
        return urlunsplit((parts.scheme, f"{hostname}{port}", parts.path, parts.query, parts.fragment))

    @staticmethod
    def _host_header(scope: Scope) -> str:
        for key, value in scope.get("headers", []):
            if key == b"host":
                return value.decode("latin-1")

        server = scope.get("server")
        if not server:
            return "localhost"

        host, port = server
        return f"{host}:{port}" if port else host
