from dulwich.repo import Repo
from pathlib import Path
from demo.lib import fs


def file_commit_timestamp(repo: Repo, path: Path) -> int:
    """Get the latest commit timestamp, in unix time format, for
    the file at `path`` in `repo`.

    Raises:
        FileNotFoundError: the given path does not exist in the
            repo or commit history
    """
    path = path.resolve().relative_to(repo.path)
    walker = repo.get_walker(paths=[bytes(path)], max_entries=1)

    try:
        entry = next(iter(walker))
    except StopIteration as exc:
        raise FileNotFoundError(path) from exc

    return entry.commit.author_time


def file_should_update(repo: Repo, src: Path, dst: Path) -> bool:
    """
    Returns `True` if the file at `src` is newer than `dst` from
    a version control perspective. That is:

        - the content of `src` and `dst` are different
        - `src` was updated after `dst`
    """
    if not dst.exists():
        return True

    if fs.file_sha(src) != fs.file_sha(dst):
        return file_commit_timestamp(repo, src) > file_commit_timestamp(repo, dst)

    return False
