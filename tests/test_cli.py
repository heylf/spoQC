import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "spoqc", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--help" in result.stdout
