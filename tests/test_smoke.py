import pytest

import local_ai_inference


def test_package_has_version() -> None:
    assert local_ai_inference.__version__ == "0.3.0"


def test_cli_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        local_ai_inference.main(["--version"])
    assert exc.value.code == 0
    assert "local-ai-inference 0.3.0" in capsys.readouterr().out
