from __future__ import annotations

import os
import subprocess

from pathlib import Path

from poetry.core.vcs.git import Git


def get_vcs(directory: Path) -> Git | None:
    working_dir = Path.cwd()
    os.chdir(str(directory.resolve()))

    vcs: Git | None

    try:
        from poetry.core.vcs.git import executable

        check_ignore = subprocess.run(
            [executable(), "check-ignore", "."],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        ).returncode

        if check_ignore == 0:
            vcs = None
        else:
            git_dir = subprocess.check_output(
                [executable(), "rev-parse", "--show-toplevel"],
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            ).strip()

            vcs = Git(Path(git_dir))

    except (subprocess.CalledProcessError, OSError, RuntimeError):
        vcs = None
    finally:
        os.chdir(str(working_dir))

    return vcs
