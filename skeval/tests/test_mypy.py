import subprocess


def test_mypy_clean():
    result = subprocess.run(["mypy", "-p", "skeval"], capture_output=True, text=True)
    assert result.returncode == 0, f"Mypy errors:\n{result.stdout}\n{result.stderr}"
