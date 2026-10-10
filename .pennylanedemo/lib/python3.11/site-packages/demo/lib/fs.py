from pathlib import Path
import shutil
import hashlib


def copy_any(src: Path, dest: Path, exist_ok: bool = False):
    """Copy `src` to `path`. If `src` is a directory, copy it recursively."""
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=exist_ok)
    else:
        shutil.copy2(src, dest)


def copy_parents(src: Path, dest: Path):
    """Copy `src` to `dest` with all parent directories."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def file_sha(path: Path) -> bytes:
    """Return the SHA256 hash of file at ``path``."""
    with open(path, "rb") as f:
        m = hashlib.sha256()
        m.update(f.read())

        return m.digest()


def clean_dir(path: Path) -> None:
    """Creates empty directory at ``path``. If it already exists, delete its contents."""
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
