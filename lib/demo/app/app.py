import typer
from demo.context import Context
from demo.lib import demo, cli, template, fs
import shutil
import logging
from typing import Annotated
import typing
import re
import json
import rich
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PLACEHOLDER_THUMBNAIL = (
    "_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_placeholder.png"
)
DEFAULT_BUILD_FORMAT = "json"
DEMO_FILENAME = "demo.py"
METADATA_FILENAME = "metadata.json"
REQUIREMENTS_FILENAME = "requirements.in"
THUMBNAIL_FILENAME = "thumbnail"
LARGE_THUMBNAIL_FILENAME = "large_thumbnail"

app = typer.Typer(
    name="demo",
    no_args_is_help=True,
    help="PennyLane demo build tool - Create, build, and manage PennyLane demos.",
)


@app.command()
def help():
    print("PennyLane demo build tool - Create, build, and manage PennyLane demos.")


@app.command()
def build(
    demo_names: Annotated[
        list[str] | None,
        typer.Argument(
            help="Names of demos to build. If not provided, build all demos."
        ),
    ] = None,
    format: Annotated[
        demo.BuildTarget, typer.Option(help="Format to build demos")
    ] = typing.cast(demo.BuildTarget, DEFAULT_BUILD_FORMAT),
    execute: Annotated[
        bool, typer.Option(help="Whether to execute demos and generate output cells")
    ] = False,
    quiet: Annotated[bool, typer.Option(help="Suppress sphinx output")] = False,
    keep_going: Annotated[
        bool, typer.Option(help="Continue if sphinx-build fails for a demo")
    ] = False,
    dev: Annotated[bool, typer.Option(help="Whether to use dev dependencies")] = False,
    venv: Annotated[str | None, typer.Option(help="Name of the virtual environment to install build dependencies")] = None,
    clear_cache: Annotated[bool, typer.Option(help="Clear the intersphinx cache")] = False,
) -> None:
    """
    Build the named demos.

    Args:
        demo_names: List of demo names to build. If None, builds all demos.
        format: Output format for the demos.
        execute: Whether to execute demos and generate output cells.
        quiet: Suppress sphinx output if True.
        keep_going: Continue building even if some demos fail.
        dev: Use development dependencies.
        venv: Name of the virtual environment to install build dependencies.
        clear_cache: Clear the intersphinx cache.
    Raises:
        typer.Exit: If build process fails.
    """
    try:
        ctx = Context()
        if clear_cache:
            fs.clean_dir(ctx.intersphinx_cache_path)
            logger.info(f"Cleared intersphinx cache at {ctx.intersphinx_cache_path}")
        demo_names = demo_names or []

        # Validate demo names exist before processing
        if demo_names:
            invalid_demos = [
                name for name in demo_names if not demo.get(ctx.demos_dir, name)
            ]
            if invalid_demos:
                logger.error(f"Demo(s) not found: {', '.join(invalid_demos)}")
                raise typer.Exit(1)

        demos = list(demo.find(ctx.demos_dir, *demo_names))

        ctx.build_dir.mkdir(exist_ok=True)

        # Add error handling for file operations
        try:
            shutil.copytree(
                ctx.repo_root / "_static", ctx.build_dir / "_static", dirs_exist_ok=True
            )
        except (OSError, shutil.Error) as e:
            logger.error(f"Failed to copy static files: {e}")
            raise typer.Exit(1)

        demo.build(
            ctx,
            demos=demos,
            target=format,
            execute=execute,
            quiet=quiet,
            keep_going=keep_going,
            dev=dev,
            venv=venv,
        )

    except Exception as e:
        logger.error(f"Build failed: {e}")
        raise typer.Exit(1)


def _slugify_title(title: str) -> str:
    """Convert a string to a slug-friendly format. This function handles CamelCase, removes special characters,
    replaces spaces and hyphens with underscores, and converts the string to lowercase.

    Examples:
    slugify("Hello World") -> "hello_world"
    slugify("CamelCaseExample") -> "camel_case_example"
    slugify("Special!@#Characters") -> "special_characters"
    slugify("Multiple   Spaces") -> "multiple_spaces"
    slugify("Hyphen-ated-Text") -> "hyphen_ated_text"
    """
    # Handle CamelCase by inserting spaces before uppercase letters
    title = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)
    # Remove special characters except spaces and hyphens
    title = re.sub(r"[^\w\s-]", "", title)
    # Replace multiple spaces/hyphens with single underscore
    title = re.sub(r"[\s-]+", "_", title)
    # Convert to lowercase and strip
    return title.strip().lower()


def _validate_title(title: str) -> str:
    """Validate and clean the demo title."""
    title = title.strip()
    if not title:
        logger.error("Title cannot be empty")
        raise typer.Exit(1)
    return title


def _validate_directory_name(name: str) -> bool:
    """Validate directory name format."""
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


def _author_format(author: str) -> dict:
    """Format author information as a dictionary."""
    return {"username": author.strip()}


def _collect_authors() -> list[dict]:
    """Collect author information from user input."""
    prompt_message = "Author's pennylane.ai username"
    authors = []

    authors.append(_author_format(typer.prompt(prompt_message)))

    while typer.confirm("Would you like to add another author?"):
        authors.append(_author_format(typer.prompt(prompt_message)))

    return authors


def _setup_thumbnails(
    demo_dir: Path, small_thumb: Path | None, large_thumb: Path | None
) -> tuple[str | None, str | None]:
    """Verify thumbnail files exist and return their paths."""
    small_thumbnail_path = None
    large_thumbnail_path = None

    if small_thumb:
        small_thumbnail_path = str(f"/{small_thumb}")

    if large_thumb:
        large_thumbnail_path = str(f"/{large_thumb}")

    return small_thumbnail_path, large_thumbnail_path

def _create_demo_files(
    demo_dir: Path,
    title: str,
    description: str,
    authors: list[str],
    small_thumbnail: str | None,
    large_thumbnail: str | None,
) -> None:
    """Create demo.py and metadata.json files."""
    try:
        demo_content = template.demo(title)
        metadata_content = template.metadata(
            title=title,
            description=description,
            authors=authors,
            thumbnail=small_thumbnail,
            large_thumbnail=large_thumbnail,
            categories=[],
        )
        demo_requirements = template.requirements()

        demo_file = demo_dir / DEMO_FILENAME
        metadata_file = demo_dir / METADATA_FILENAME
        requirements_file = demo_dir / REQUIREMENTS_FILENAME

        demo_file.write_text(demo_content)
        metadata_file.write_text(json.dumps(metadata_content, indent=2))
        requirements_file.write_text(demo_requirements)
        logger.info(
            f"Created demo files in {demo_dir.relative_to(Context().repo_root)}"
        )
    except (OSError, json.JSONEncodeError) as e:
        logger.error(f"Failed to create demo files: {e}")
        raise typer.Exit(1)


@app.command()
def new():
    """Create a new demo."""
    try:
        ctx = Context()

        # Get and validate title
        title = typer.prompt("Title")
        title = _validate_title(title)

        # Get directory name with validation
        name_default = _slugify_title(title)
        while True:
            name = typer.prompt("Custom directory name", name_default)

            if not _validate_directory_name(name):
                logger.error(
                    "Directory name can only contain letters, numbers, hyphens, and underscores"
                )
                continue

            if demo.get(ctx.demos_dir, name):
                logger.warning(
                    f"Demo '{name}' already exists, please choose a different name."
                )
                continue
            break

        # Get other metadata
        description = typer.prompt("Description", "")
        authors = _collect_authors()

        # Get thumbnails
        small_thumbnail = cli.prompt_path("Thumbnail image", PLACEHOLDER_THUMBNAIL)
        large_thumbnail = cli.prompt_path("Large thumbnail image")

        # Create demo directory
        demo_dir = ctx.demos_dir / name
        try:
            demo_dir.mkdir(exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create demo directory: {e}")
            raise typer.Exit(1)

        # Setup thumbnails
        small_thumb_path, large_thumb_path = _setup_thumbnails(
            demo_dir, small_thumbnail, large_thumbnail
        )

        # Create demo files
        _create_demo_files(
            demo_dir, title, description, authors, small_thumb_path, large_thumb_path
        )

        rich.print(
            f"Created new demo in [bold]{demo_dir.relative_to(ctx.repo_root)}[/bold]"
        )

    except KeyboardInterrupt:
        logger.info("Demo creation cancelled by user")
        raise typer.Exit(0)
    except Exception as e:
        logger.error(f"Failed to create demo: {e}")
        raise typer.Exit(1)
