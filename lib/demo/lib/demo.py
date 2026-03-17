from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Sequence, Iterator
from packaging.version import Version
import shutil
from demo.lib import fs, cmds
from demo.lib.virtual_env import Virtualenv
import os
import sys
from logging import getLogger
import subprocess
from enum import Enum
import functools
import requirements
import json
import lxml.html
from demo.context import Context
import sphobjinv as soi


logger = getLogger("demo")


class BuildTarget(Enum):
    """Sphinx-build targets."""

    HTML = "html"
    JSON = "json"


@dataclass
class Demo:
    """Represents a demo and its metadata."""

    CORE_DEPENDENCIES = frozenset(
        (
            "aiohttp",
            "fsspec",
            "h5py",
            "jax",
            "jaxlib",
            "matplotlib",
            "numpy",
            "pennylane",
            "pennylane-lightning",
            "pennylane-catalyst",
        )
    )
    """Dependencies installed for every demo that do not
    need to be specified by `requirements.in`."""

    name: str
    path: Path

    @property
    def py_file(self) -> Path:
        """The python file containing this demo's code and
        markup."""
        return self.path / "demo.py"

    @property
    def metadata_file(self) -> Path:
        """Metadata for this demo."""
        return self.path / "metadata.json"

    @property
    def requirements_file(self) -> Path | None:
        """Path to a requirements file containing
        unversioned dependencies for this demo."""
        if (path := self.path / "requirements.in").exists():
            return path

        return None

    @property
    def resources(self) -> Sequence[Path]:
        """Other files in the demo's directory."""
        return tuple(
            p
            for p in self.path.iterdir()
            if p not in {self.py_file, self.metadata_file, self.requirements_file}
        )

    @property
    def executable_stable(self) -> bool:
        """Whether this demo can be executed for stable builds."""
        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata.get("executable_stable", self.name.startswith("tutorial_"))

    @property
    def executable_latest(self) -> bool:
        """Whether this demo can be executed for dev builds."""
        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata.get("executable_latest", self.name.startswith("tutorial_"))

    @functools.cached_property
    def requirements(self) -> frozenset[str]:
        if not (path := self.requirements_file):
            return self.CORE_DEPENDENCIES

        reqs = set(self.CORE_DEPENDENCIES)
        with open(path, "r") as f:
            for req in requirements.parse(f):
                reqs.discard(req.name)
                reqs.add(req.line)

        return frozenset(reqs)


def get(search_dir: Path, name: str) -> Demo | None:
    """Get demo with `name`, if it exists."""
    demo = Demo(name=name, path=search_dir / name)

    if not demo.py_file.exists():
        return None

    return demo


def find(search_dir: Path, *names: str) -> Iterator[Demo]:
    """Find demos with given names in `search_dir`."""
    if not names:
        yield from (
            Demo(name=demo_dir.name, path=demo_dir.resolve())
            for demo_dir in search_dir.iterdir()
            if demo_dir.is_dir()
        )

        return

    for name in set(names):
        demo_dir = search_dir / name
        if not (demo_dir / "demo.py").exists():
            raise ValueError(f"No demo exists with name '{name}'")

        yield Demo(name=name, path=demo_dir.resolve())


def search(search_dir: Path, pattern: str) -> Iterator[str]:
    """Yield demo names in `search_dir` matching `pattern`."""
    for path in search_dir.glob(pattern=pattern):
        if (path / "demo.py").exists():
            yield path.name


def build(
    ctx: Context,
    demos: Sequence[Demo],
    target: BuildTarget,
    execute: bool,
    quiet: bool = False,
    keep_going: bool = False,
    dev: bool = False,
    venv: str | None = None,
) -> None:
    """Build the provided demos using 'sphinx-build', optionally
    executing them to generate plots and cell outputs.

    Args:
        sphinx_dir: The directory containing the sphinx conf.py file
        build_dir: Build directory
        venv_path: Path to virtual environment into which build and
            dependencies will be installed
        demos: List of demos to build
        target: The target build format
        execute: Whether to execute demos
    """
    failed: list[str] = []
    done = 0
    logger.info("Building %d demos", len(demos))

    for demo in demos:
        execute_demo = execute and (demo.executable_latest if dev else demo.executable_stable)
        done += 1
        logger.info(
            "Building '%s' (%d/%d), execute=%s",
            demo.name,
            done,
            len(demos),
            execute_demo,
        )
        build_venv = Virtualenv(ctx.repo_root / venv) if venv else Virtualenv(ctx.build_venv_path)
        cmds.pip_install(
            build_venv.python,
            requirements=ctx.build_requirements_file,
            quiet=quiet,
        )
        try:
            _build_demo(
                ctx,
                build_venv=build_venv,
                target=target,
                execute=execute_demo,
                demo=demo,
                package=target is BuildTarget.JSON,
                quiet=quiet,
                dev=dev,
            )
        except subprocess.CalledProcessError as exc:
            if not keep_going:
                raise exc

            failed.append(demo.name)
            logger.error("Build failed for demo '%s'", demo.name)
            if quiet:
                if (
                    error_summary := _find_sphinx_gallery_execution_error(exc.stdout)
                ) is None:
                    error_summary = exc.stdout

                logger.error("%s", error_summary)

    if failed:
        raise RuntimeError(f"Failed to build {len(failed)} demos", failed)

    # If we built the HTML output, gather and merge the objects.inv files
    if target is BuildTarget.HTML:
        logger.info("Building the master objects.inv file.")

        inventory = soi.Inventory()
        inventory.project = "PennyLane"

        for demo in demos:
            logger.info("Loading objects.inv for '%s'", demo.name)
            demo_inv = soi.Inventory(
                ctx.build_dir / f"{demo.name}_objects.inv"
            )

            # Only add entries that don't already exist in the merged inventory file
            for entry in demo_inv.objects:
                if entry not in inventory.objects:
                    logger.info("Appending inventory object '%s'", entry.name)
                    inventory.objects.append(entry)

        logger.info("Writing the master objects.inv file to %s.", ctx.build_dir)
        text = inventory.data_file(contract=True)
        ztext = soi.compress(text)
        soi.writebytes(ctx.build_dir / "objects.inv", ztext)


def generate_requirements(
    ctx: Context, demo: Demo, dev: bool, output_file: Path
) -> None:
    constraints = [ctx.build_requirements_file]
    if dev:
        constraints.append(ctx.dev_constraints_file)
    else:
        constraints.append(ctx.stable_constraints_file)

    requirements_in = [ctx.core_requirements_file]
    if demo.requirements_file:
        requirements_in.append(demo.requirements_file)

    cmds.pip_compile(
        sys.executable,
        output_file,
        *requirements_in,
        constraints_files=constraints,
        quiet=False,
        prerelease=False,
    )


def _build_demo(
    ctx: Context,
    build_venv: Virtualenv,
    demo: Demo,
    target: BuildTarget,
    execute: bool,
    package: bool,
    quiet: bool,
    dev: bool,
):
    out_dir = ctx.repo_root / "demos"
    fs.clean_dir(out_dir)

    generate_requirements(ctx, demo, dev, out_dir / "requirements.txt")
    if execute:
        cmds.pip_install(
            build_venv.python,
            "--upgrade",
            requirements=out_dir / "requirements.txt",
            quiet=quiet,
        )
        # If dev, we need to re-install the latest Catalyst, then Lightning, then PennyLane
        # in that order, regardless of conflicts/warnings. 
        if dev:
            base_install_args = [
                build_venv.python,
                "--extra-index-url",
                "https://test.pypi.org/simple/",
            ]
            with open(ctx.plc_dev_constraints_file) as f:
                constraints = json.load(f)
            for package in constraints:
                logger.info("Getting versions for %s", package["name"])
                package_versions = cmds.pip_get_versions(build_venv.python, package["name"], index_url="https://test.pypi.org/simple/")
                # These are returned newest to oldest, so we can break when we find the first match
                for version in package_versions:
                    if Version(version).release == Version(package["version"]).release:
                        logger.info("Installing %s==%s", package["name"], version)
                        install_args = base_install_args.copy()
                        if package["name"] != "pennylane-catalyst":
                            install_args.append("--force-reinstall")
                            install_args.append("--no-deps")
                        install_args.append(f"{package['name']}=={version}")
                        cmds.pip_install(*install_args, quiet=quiet, pre=True)
                        break

            # Need to reinstall the demo's requirements file to ensure the correct versions are installed
            if demo.requirements_file:
                cmds.pip_install(
                    build_venv.python,
                    "--upgrade",
                    requirements=demo.requirements_file,
                    quiet=quiet,
                )

    elif dev:
        # Need latest version of PennyLane to build, whether or not we're executing
        cmds.pip_install(
            build_venv.python,
            "--upgrade",
            "git+https://github.com/PennyLaneAI/pennylane.git#egg=pennylane",
            quiet=quiet,
        )

    stage_dir = ctx.build_dir / "demonstrations"
    fs.clean_dir(stage_dir)
    # Need a 'GALLERY_HEADER' file for sphinx-gallery
    with open(stage_dir / "GALLERY_HEADER.rst", "w"):
        pass

    shutil.copy2(demo.py_file, (stage_dir / demo.name).with_suffix(".py"))
    for resource in demo.resources:
        fs.copy_any(resource, (stage_dir / resource.name))

    cmd = [
        str(build_venv.path / "bin" / "sphinx-build"),
        "-b",
        target.value,
    ]
    if not execute:
        cmd.extend(("-D", "plot_gallery=0"))

    cmd.extend((str(ctx.repo_root), str(ctx.build_dir / target.value)))
    sphinx_env = os.environ | {
        "DEMO_STAGING_DIR": str(stage_dir.resolve()),
        "GALLERY_OUTPUT_DIR": str(out_dir.resolve().relative_to(ctx.repo_root)),
        # Make sure demos can find scripts installed in the build venv
        "PATH": f"{os.environ['PATH']}:{build_venv.path / 'bin'}",
        "CURRENT_DEMO": str(demo.name),
        "DEV": str(dev),
    }
    if quiet:
        stdout, stderr, text = subprocess.PIPE, subprocess.STDOUT, True
    else:
        stdout, stderr, text = None, None, None

    subprocess.run(
        cmd, env=sphinx_env, stdout=stdout, stderr=stderr, text=text
    ).check_returncode()

    if package:
        _package_demo(
            demo,
            ctx.build_dir / "pack",
            ctx.repo_root / "_static",
            ctx.build_dir / target.value,
            out_dir,
        )

    # Move the objects.inv file so we can merge them once all the demos are built
    if target is BuildTarget.HTML:
        fs.copy_any(ctx.build_dir / "html/objects.inv", ctx.build_dir / f"{demo.name}_objects.inv")


def _package_demo(
    demo: Demo,
    pack_dir: Path,
    static_dir: Path,
    sphinx_output: Path,
    sphinx_gallery_output: Path,
):
    """Package a demo into a .zip file for distribution.

    Args:
        demo: The demo to package
        pack_dir: The directory in which to place the packaged demo
        static_dir: The /static directory in the repo root
        sphinx_output: The directory containing the sphinx output
        sphinx_gallery_output: The directory containing files genreated by
            sphinx-gallery
    """
    dest = pack_dir / demo.name
    fs.clean_dir(dest)

    with open(
        (sphinx_output / "demos" / demo.name).with_suffix(".fjson"), "r"
    ) as f:
        html_body = json.load(f)["body"]

    asset_paths: set[tuple[Path, str]] = set()
    html_body: str = lxml.html.rewrite_links(
        html_body,
        functools.partial(
            _link_rewriter, static_dir, sphinx_output / "_images", asset_paths
        ),
    )
    with open(dest / "body.html", "w") as f:
        f.write(html_body)

    for asset, asset_dest in asset_paths:
        try:
            fs.copy_parents(asset, Path(dest, "_assets", asset_dest))
        except FileNotFoundError:
            logger.warning("Could not find asset '%s' for demo '%s'", asset, demo.name)
            continue

    shutil.copy(
        (sphinx_gallery_output / demo.name).with_suffix(".ipynb"), dest / "demo.ipynb"
    )
    shutil.copy(demo.metadata_file, dest / "metadata.json")
    shutil.copy(demo.py_file, dest / "demo.py")
    shutil.copy(sphinx_gallery_output / "requirements.txt", dest / "requirements.txt")
    for resource in demo.resources:
        fs.copy_any(resource, dest / resource.relative_to(demo.path))

    with open(demo.metadata_file, "r") as f:
        metadata = json.load(f)

    for preview_image in metadata["previewImages"]:
        if (uri := preview_image["uri"]).startswith("/_static/"):
            src = static_dir / uri.removeprefix("/_static/")
            path = PurePosixPath(
                "_assets", "thumbnails", preview_image["type"]
            ).with_suffix(src.suffix)
            preview_image["uri"] = path.as_posix()
            fs.copy_parents(src, dest / path)

    for hardware in metadata.get("hardware", []):
        if (uri := hardware["logo"]).startswith("/_static/"):
            src = static_dir / uri.removeprefix("/_static/")
            path = PurePosixPath("_assets", "logos", Path(src).name)
            hardware["logo"] = path.as_posix()
            fs.copy_parents(src, dest / path)

    with open(dest / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    shutil.make_archive(
        base_name=str(pack_dir / dest.name), format="zip", base_dir=dest, root_dir=dest
    )


def _link_rewriter(
    static_dir: Path, image_dir: Path, asset_paths: set[tuple[Path, str]], link: str
):
    if "_images/" in link:
        _, path = link.split("_images/", maxsplit=2)
        asset_paths.add((image_dir / path, f"images/{path}"))
        return f"_assets/images/{path}"
    elif "_static/" in link:
        _, path = link.split("_static/", maxsplit=2)
        asset_paths.add((static_dir / path, f"static/{path}"))
        return f"_assets/static/{path}"

    return link


def _find_sphinx_gallery_execution_error(stdout: str) -> str | None:
    """Parse the error section from sphinx-gallery output.

    Returns `None` if the error section could not be found."""
    i = stdout.find(
        "Here is a summary of the problems encountered when running the examples:"
    )
    if i != -1:
        return stdout[i:]

    return None
