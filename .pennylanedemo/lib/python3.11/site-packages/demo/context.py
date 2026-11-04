from dulwich.repo import Repo
from pathlib import Path
import functools


class Context:
    """Context for CLI commands."""

    @property
    def repo_root(self) -> Path:
        """Absolute path to repository root."""
        return Path(self.repo.path).resolve()

    @property
    def demos_dir(self) -> Path:
        """Path to the content dir, relative to the current
        working directory."""
        return self.repo_root / "demonstrations_v2"

    @property
    def build_dir(self) -> Path:
        """Path to the build directory."""
        return self.repo_root / "_build"

    @property
    def build_venv_path(self) -> Path:
        """Path to virtual environment for building demos."""
        return self.repo_root / ".venv-build"

    @property
    def intersphinx_cache_path(self) -> Path:
        """Path to intersphinx cache."""
        return self.repo_root / ".intersphinx_cache"

    @functools.cached_property
    def cwd(self) -> Path:
        """Current working directory of the process."""
        return Path.cwd().resolve()

    @functools.cached_property
    def repo(self) -> Repo:
        """dulwich ``Repo`` object for the repo."""
        return Repo.discover()

    @property
    def core_requirements_file(self) -> Path:
        return self.repo_root / "dependencies" / "requirements-core.in"

    @property
    def stable_constraints_file(self) -> Path:
        return self.repo_root / "dependencies" / "constraints-stable.txt"

    @property
    def dev_constraints_file(self) -> Path:
        return self.repo_root / "dependencies" / "constraints-dev.txt"

    @property
    def plc_dev_constraints_file(self) -> Path:
        return self.repo_root / "dependencies" / "constraints-plc-dev.json"

    @property
    def build_requirements_file(self) -> Path:
        return self.repo_root / "dependencies" / "requirements-build.txt"
