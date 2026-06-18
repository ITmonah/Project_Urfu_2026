import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = "likip3/AI-Models"
DEFAULT_MODEL_RELEASE_TAG = "v1"


@dataclass(frozen=True)
class ModelAssetSpec:
    label: str
    path_env_var: str
    default_path: Path
    asset_name: str
    url_env_var: str


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def model_cache_dir() -> Path:
    return resolve_path(os.getenv("KGO_MODEL_CACHE_DIR", ROOT_DIR / ".model_cache"))


def github_release_asset_url(asset_name: str) -> str:
    repo = os.getenv("KGO_MODEL_REPO", DEFAULT_MODEL_REPO).strip("/")
    tag = os.getenv("KGO_MODEL_RELEASE_TAG", DEFAULT_MODEL_RELEASE_TAG)
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def model_asset_url(spec: ModelAssetSpec) -> str:
    direct_url = os.getenv(spec.url_env_var)
    if direct_url:
        return direct_url

    base_url = os.getenv("KGO_MODEL_ASSETS_BASE_URL")
    if base_url:
        return f"{base_url.rstrip('/')}/{spec.asset_name}"

    return github_release_asset_url(spec.asset_name)


def resolve_model_asset_path(spec: ModelAssetSpec) -> Path:
    env_path = os.getenv(spec.path_env_var)
    if env_path:
        return resolve_path(env_path)

    default_path = resolve_path(spec.default_path)
    if default_path.exists():
        return default_path

    return model_cache_dir() / spec.asset_name


def model_asset_status(spec: ModelAssetSpec) -> dict[str, str | bool]:
    path = resolve_model_asset_path(spec)
    return {
        "label": spec.label,
        "env_var": spec.path_env_var,
        "url_env_var": spec.url_env_var,
        "path": str(path),
        "source_url": model_asset_url(spec),
        "available": path.exists(),
    }


def _content_length(response) -> int | None:
    value = response.headers.get("Content-Length") if hasattr(response, "headers") else None
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _format_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.1f} MB"


def _log_progress(message: str) -> None:
    print(message, flush=True, file=sys.stderr)


def ensure_model_asset(spec: ModelAssetSpec, *, log_progress: bool = False) -> Path:
    destination = resolve_model_asset_path(spec)
    if destination.exists():
        if log_progress:
            _log_progress(f"[model-assets] {spec.asset_name} already exists: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = model_asset_url(spec)
    temp_destination = destination.with_suffix(destination.suffix + ".part")

    headers = {}
    token = os.getenv("KGO_MODEL_AUTH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    try:
        if log_progress:
            _log_progress(f"[model-assets] Downloading {spec.asset_name} from {url}")
        with urlopen(request, timeout=60) as response:
            total_size = _content_length(response)
            downloaded_size = 0
            last_log_at = time.monotonic()
            with temp_destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded_size += len(chunk)
                    if log_progress:
                        now = time.monotonic()
                        if now - last_log_at >= 2:
                            if total_size:
                                percent = downloaded_size / total_size * 100
                                _log_progress(
                                    "[model-assets] "
                                    f"{spec.asset_name}: {_format_size(downloaded_size)} / "
                                    f"{_format_size(total_size)} ({percent:.1f}%)"
                                )
                            else:
                                _log_progress(
                                    f"[model-assets] {spec.asset_name}: {_format_size(downloaded_size)} downloaded"
                                )
                            last_log_at = now
        temp_destination.replace(destination)
        if log_progress:
            _log_progress(f"[model-assets] Finished {spec.asset_name}: {destination}")
    except (OSError, URLError) as exc:
        if temp_destination.exists():
            temp_destination.unlink()
        raise RuntimeError(
            f"Failed to download {spec.label} checkpoint from {url}: {exc}"
        ) from exc

    return destination
