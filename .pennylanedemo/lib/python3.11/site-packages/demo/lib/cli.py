import typer
from pathlib import Path
import rich


def prompt_path(prompt: str, default_path: str | None = "") -> Path | None:
    """Prompt user for a file path.

    Args:
        prompt: Prompt text

    Returns:
        Path: resolved path
        None: User provided no inputs
    """

    path = typer.prompt(prompt, default=default_path)
    if not path:
        return None

    path = Path(path).expanduser()
    if not path.exists():
        rich.print(f"File [red][bold]{path}[/bold][/red] does not exist.")

        return prompt_path(prompt)

    return path
