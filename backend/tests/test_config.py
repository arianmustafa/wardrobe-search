"""Settings: API key resolution (inline vs. file) and demo-mode detection."""
from app.config import Settings


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_no_key_means_demo_mode():
    assert make_settings(gemini_api_key=None).use_real_embeddings is False


def test_blank_key_means_demo_mode():
    assert make_settings(gemini_api_key="   ").use_real_embeddings is False


def test_inline_key_enables_real_embeddings():
    assert make_settings(gemini_api_key="abc123").use_real_embeddings is True


def test_key_read_from_file(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("  file-key\n")
    settings = make_settings(gemini_api_key_file=str(key_file))
    assert settings.gemini_api_key == "file-key"
    assert settings.use_real_embeddings is True


def test_inline_key_wins_over_file(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-key")
    settings = make_settings(gemini_api_key="inline-key", gemini_api_key_file=str(key_file))
    assert settings.gemini_api_key == "inline-key"


def test_blank_inline_key_falls_back_to_file(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-key")
    settings = make_settings(gemini_api_key="  ", gemini_api_key_file=str(key_file))
    assert settings.gemini_api_key == "file-key"


def test_missing_key_file_leaves_demo_mode(tmp_path):
    settings = make_settings(gemini_api_key_file=str(tmp_path / "nope.txt"))
    assert not settings.gemini_api_key
    assert settings.use_real_embeddings is False
