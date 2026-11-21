import local_ai_inference


def test_package_imports_with_version() -> None:
    assert local_ai_inference.__version__ == "0.0.0"


def test_cli_entry_point_prints_version(capsys) -> None:
    local_ai_inference.main()
    assert "local-ai-inference 0.0.0" in capsys.readouterr().out
