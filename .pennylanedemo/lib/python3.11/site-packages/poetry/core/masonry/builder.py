from __future__ import annotations

import warnings

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

    from poetry.core.poetry import Poetry


warnings.warn(
    "poetry.core.masonry.builder is deprecated. Its functionality has been moved"
    "from poetry-core to poetry (poetry.console.commands.build).",
    DeprecationWarning,
    stacklevel=2,
)


class Builder:
    def __init__(self, poetry: Poetry) -> None:
        from poetry.core.masonry.builders.sdist import SdistBuilder
        from poetry.core.masonry.builders.wheel import WheelBuilder

        self._poetry = poetry

        self._formats = {
            "sdist": SdistBuilder,
            "wheel": WheelBuilder,
        }

    def build(
        self,
        fmt: str,
        executable: str | Path | None = None,
        *,
        target_dir: Path | None = None,
    ) -> None:
        if fmt in self._formats:
            builders = [self._formats[fmt]]
        elif fmt == "all":
            builders = list(self._formats.values())
        else:
            raise ValueError(f"Invalid format: {fmt}")

        for builder in builders:
            builder(self._poetry, executable=executable).build(target_dir)
