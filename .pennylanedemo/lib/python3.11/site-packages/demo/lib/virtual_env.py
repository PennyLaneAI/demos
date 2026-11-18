from pathlib import Path
import subprocess
import sys


class Virtualenv:
    """Interface to a Python virtual environment."""

    def __init__(self, path: Path, *, quiet: bool = False):
        """
        Get python virtual env, creating it if it does not exist.
        Args:
            path: Path to virtual env directory. Will be initialized
                if it does not exist.
            quiet: When True, pass ``--quiet`` to ``pip install uv``.
        """
        self.path = path.resolve()
        self._quiet = quiet
        self._init()

    @property
    def python(self) -> Path:
        """Path to the python executable in this virtual env."""
        return self.path / "bin" / "python"

    def _init(self):
        """Initialize a virtual environment."""
        self.path.parent.mkdir(exist_ok=True)

        subprocess.run([sys.executable, "-m", "venv", "--clear", "--upgrade-deps", self.path]).check_returncode()
        uv_install = [str(self.python), "-m", "pip", "install"]
        if self._quiet:
            uv_install.append("--quiet")
        uv_install.append("uv")
        subprocess.run(uv_install).check_returncode()
