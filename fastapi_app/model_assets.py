import os
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = "likip3/AI_Models"
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


def ensure_model_asset(spec: ModelAssetSpec) -> Path:
    destination = resolve_model_asset_path(spec)
    if destination.exists():
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
        with urlopen(request, timeout=60) as response:
            with temp_destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        temp_destination.replace(destination)
    except (OSError, URLError) as exc:
        if temp_destination.exists():
            temp_destination.unlink()
        raise RuntimeError(
            f"Failed to download {spec.label} checkpoint from {url}: {exc}"
        ) from exc

    return destination
