from pathlib import Path

from fastapi_app import model_assets
from fastapi_app.model_assets import ModelAssetSpec


def make_spec(default_path: Path) -> ModelAssetSpec:
    return ModelAssetSpec(
        label="YOLO",
        path_env_var="KGO_TEST_CHECKPOINT",
        default_path=default_path,
        asset_name="model.pt",
        url_env_var="KGO_TEST_URL",
    )


def test_explicit_checkpoint_path_has_priority(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "local" / "model.pt"
    monkeypatch.setenv("KGO_TEST_CHECKPOINT", str(checkpoint_path))

    assert model_assets.resolve_model_asset_path(make_spec(tmp_path / "default.pt")) == checkpoint_path


def test_resolves_missing_asset_to_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("KGO_TEST_CHECKPOINT", raising=False)
    monkeypatch.setenv("KGO_MODEL_CACHE_DIR", str(tmp_path / "cache"))

    path = model_assets.resolve_model_asset_path(make_spec(tmp_path / "missing.pt"))

    assert path == tmp_path / "cache" / "model.pt"


def test_existing_default_checkpoint_is_used_before_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("KGO_TEST_CHECKPOINT", raising=False)
    monkeypatch.setenv("KGO_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    default_path = tmp_path / "dvc" / "model.pt"
    default_path.parent.mkdir()
    default_path.write_bytes(b"model")

    path = model_assets.resolve_model_asset_path(make_spec(default_path))

    assert path == default_path


def test_builds_github_release_url(monkeypatch, tmp_path):
    monkeypatch.delenv("KGO_TEST_URL", raising=False)
    monkeypatch.delenv("KGO_MODEL_ASSETS_BASE_URL", raising=False)
    monkeypatch.setenv("KGO_MODEL_REPO", "likip3/AI-Models")
    monkeypatch.setenv("KGO_MODEL_RELEASE_TAG", "v1")

    url = model_assets.model_asset_url(make_spec(tmp_path / "missing.pt"))

    assert url == "https://github.com/likip3/AI-Models/releases/download/v1/model.pt"


def test_direct_url_env_has_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("KGO_TEST_URL", "https://example.com/custom.pt")

    assert model_assets.model_asset_url(make_spec(tmp_path / "missing.pt")) == "https://example.com/custom.pt"


def test_downloads_missing_asset(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self):
            self.chunks = [b"abc", b"123", b""]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return self.chunks.pop(0)

    monkeypatch.setenv("KGO_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(model_assets, "urlopen", lambda request, timeout: FakeResponse())

    path = model_assets.ensure_model_asset(make_spec(tmp_path / "missing.pt"))

    assert path.read_bytes() == b"abc123"
