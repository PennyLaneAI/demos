# CLI Tool

## Overview

The **CLI tool** streamlines the creation, testing, and building of PennyLane demonstrations. It provides an intuitive command-line interface to simplify the authoring process and maintain consistency across demonstration projects.

## Table of Contents

*   [Overview](#overview)
*   [Installation](#installation)
*   [Commands](#commands)
    *   [`help`](#help)
    *   [`new`](#new)
    *   [`build`](#build)
    *   [Build Flags](#build-flags)
*   [Viewing Build Outputs](#viewing-build-outputs)
*   [Support](#support)
*   [License](#license)


## Installation

To install the CLI, you will need Python version `3.11` or later. We highly recommend using a virtual environment to install the CLI tool to manage dependencies effectively.

### MacOS/Linux Installation Steps

```bash
# Step 1. Create a virtual environment (optional but highly recommended)
python3.11 -m venv .venv

# Step 2. Activate the virtual environment
source .venv/bin/activate

# Step 3. Install the CLI tool
# This command assumes you are in the root directory of this repository.
pip install .

# Step 4. Verify the installation
demo

PennyLane demo build tool - Create, build, and manage PennyLane demos.

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                     │
│ --show-completion             Show completion for the current shell, to copy it or customize the            │
│                               installation.                                                                 │
│ --help                        Show this message and exit.                                                   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ help                                                                                                        │
│ build     Build the named demos.                                                                            │
│ new       Create a new demo.                                                                                │                        │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

### Windows Installation Steps

```bash
# Step 1. Create a virtual environment (optional but highly recommended)
python -m venv .venv

# Step 2. Activate the virtual environment
.\.venv\Scripts\activate

# Step 3. Install the CLI tool
# This command assumes you are in the root directory of this repository.
pip install .

# Step 4. Verify the installation
demo

PennyLane demo build tool - Create, build, and manage PennyLane demos.                                                                                                               

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                     │
│ --show-completion             Show completion for the current shell, to copy it or customize the            │
│                               installation.                                                                 │
│ --help                        Show this message and exit.                                                   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ help                                                                                                        │
│ build     Build the named demos.                                                                            │
│ new       Create a new demo.                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Commands

### `help`

Displays comprehensive help information for the CLI tool, including a list of available commands and their usage.

```bash
demo help
```


### `new`

Creates a new PennyLane demonstration project within the `demonstrations_v2` directory.

```bash
demo new
```

Upon executing this command, you will be prompted to provide the following details interactively:

*   **Title**: A human-readable title for the demonstration.
*   **Custom directory name**: A unique directory name (used as demonstration slug on the [PL.ai](https://pennylane.ai/) website.
*   **Description**: A concise explanation of the demonstration's purpose or functionality.
*   **Author(s)**: The PennyLane handle(s) of the author(s). If there are multiple authors, you will be prompted to enter each one individually. (\*Note\*: If you do not have a PennyLane handle, you will need to [create one](https://auth.cloud.pennylane.ai/u/signup?state=hKFo2SBqQzM4RlJmNDJZdzNjX0UwbHpYYVU2a012eUlTWDZBd6Fur3VuaXZlcnNhbC1sb2dpbqN0aWTZIHZaVUM2TGhRNjVtM3YtWjhjWGZiaTc0T1ZqTW16ZWVGo2NpZNkgU1hka2hOc2lMVDBHZHJPVEZBUjJnSjV0cThvR1ZjZzM) before authoring a demonstration.)
*   **Thumbnail**: (Optional) The path to a thumbnail image. This file must be located in `/_static/demo_thumbnails/regular_demo_thumbnails/`. You may leave this field blank and add the image later.
*   **Large Thumbnail**: (Optional) The path to a larger thumbnail image, also to be placed in `/_static/demo_thumbnails/large_demo_thumbnails/`. This can be left blank and added later.

Once all required inputs are provided, a new subdirectory will be created under `demonstrations_v2`, named according to the specified demo name. This new directory will include the following essential files:

*   `demo.py`: The main Python script containing the demonstration's code.
*   `metadata.json`: A JSON file storing descriptive metadata about the demonstration.
*   `requirements.in`: A file listing Python dependencies specifically required by this demonstration.


### `build`

Compiles one or more demonstrations, preparing them for local execution or packaging.

```bash
# Provide demo names (directories) separated by spaces.
# If no demo names are provided, all demonstrations within `demonstrations_v2` will be built.

# Example: Building all demonstrations
demo build

# Example: Building a specific demonstration
demo build my_demo

# Example: Building multiple demonstrations
demo build demo_one demo_two
```

#### Build Flags:

##### `--execute`

Executes the specified demos using Sphinx Gallery. This flag is effective only for demos whose metadata fields `executable_stable` (uses the most recent release versions of packages in the PennyLane ecosystem) or `executable_latest` (uses the latest development versions of the PennyLane ecosystem) are set to "true". Historically, directory names prefixed with `tutorial__` marked them as executable tutorials. This is the fallback behaviour if either of the aforementioned metadata fields are missing for that demonstration.

```bash
demo build --execute
```

##### `--format <html|json>`

Specifies the desired output format for the built demonstration. Options include `html` for web output or `json` (default) for structured data.

```bash
# Output as HTML
demo build --format html demo_name

# Output as JSON
demo build --format json demo_name
```

##### `--no-dev`

Instructs the build process to use stable (release) versions of [PennyLane](https://docs.pennylane.ai/), [Catalyst](https://docs.pennylane.ai/projects/catalyst), [official plugins](https://pennylane.ai/devices), and other core dependencies. This is the default build behaviour.

```bash
demo build --no-dev demo_name
```

##### `--dev`

Instructs the build process to use the latest development (unreleased) versions of [PennyLane](https://github.com/pennylaneai/pennylane), [Catalyst](https://docs.pennylane.ai/projects/catalyst/en/latest/), [official plugins](https://pennylane.ai/devices), and other core dependencies.

```bash
demo build --dev demo_name
```

##### `--venv`

Specify a new or existing virtual environment to use for installing the demo dependencies. If omitted, the defaul virtual environment name of `.venv-build` will be used.

```bash
demo build --venv .my_env
```


## Viewing Build Outputs

You can view the demonstration in any web browser. Upon successful completion of the build, the generated files will be located in the `_build/html/demos/<demo_name>/` directory if you built html, or in `_build/pack/<demo_name>/` if you built json. Note that any changes made to the source files will require a rebuild to be reflected in the output. Also note that the majority of the styling, including rendering of any math, will not be included in the local build.

## Support

*   **Source Code:** https://github.com/PennyLaneAI/demos
*   **Issue Tracker:** https://github.com/PennyLaneAI/demos/issues

For any issues or questions, please utilize our GitHub issue tracker.


## License

The materials and demonstrations provided in this repository are **free** and **open source**, distributed under the Apache License, Version 2.0.

Please note, the file `custom_directives.py` is available under the BSD 3-Clause License, with copyright © 2017, PyTorch contributors.
