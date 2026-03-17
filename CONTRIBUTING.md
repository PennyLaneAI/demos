# Contributing to PennyLane Demos

This document provides comprehensive guidelines for contributing to the PennyLane demonstrations repository. It covers the entire process, from setting up your environment and creating new demos to managing dependencies and building your contributions for review.

## Table of Contents

- [Contributing to PennyLane Demos](#contributing-to-pennylane-demos)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Cloning or Forking the Repository](#cloning-or-forking-the-repository)
    - [Creating a New Demo](#creating-a-new-demo)
      - [Using the CLI Tool](#using-the-cli-tool)
      - [Manually Creating a Demo](#manually-creating-a-demo)
  - [Tutorial Content Guidelines](#tutorial-content-guidelines)
  - [General Guidelines](#general-guidelines)
  - [Image Guidelines](#image-guidelines)
    - [File Size and Format](#file-size-and-format)
    - [Dimensions](#dimensions)
  - [Metadata Guidelines](#metadata-guidelines)
    - [Example `metadata.json` Structure](#example-metadatajson-structure)
    - [Validate Your Metadata File](#validate-your-metadata-file)
  - [Dependency Management](#dependency-management)
  - [Building and Testing Locally](#building-and-testing-locally)
    - [Building a Demo in HTML Format](#building-a-demo-in-html-format)
    - [Building a Demo in JSON Format](#building-a-demo-in-json-format)
  - [Support](#support)
  - [License](#license)

## Getting Started

### Cloning or Forking the Repository

To contribute to PennyLane demonstrations, begin by forking and cloning this repository. All contributions should be made by opening a pull request against the `master` branch (for stable versions) or the `dev` branch (for the latest features from PennyLane, Catalyst, and other plugins).

### Creating a New Demo

There are multiple ways to create a new demo. The recommended method is to use the CLI tool, which provides a structured way to set up your demo environment and ensures that all necessary files are created.

#### Using the CLI Tool

To create a new demonstration, use the CLI tool's [`new` command](/documentation/demo-cli.md#new). This command will guide you through the initial setup. You will be prompted to provide a title, a custom directory name, a description, and the author's username. Optionally, you can add thumbnail images for the demo.

To make a demo executable, set the `executable_stable` or `executable_latest` flag to `true` in the metadata.json file.

```bash
❯ demo new
Title: Your demo title
Custom directory name [your_demo_title]: your_demo_directory_name
Description []: A description of the demo you are creating
Author's pennylane.ai username: author_username
Would you like to add another author? [y/N]: n
Thumbnail image [_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_placeholder.png]: 
Large thumbnail image []: 
```

#### Manually Creating a Demo

If you prefer to set up your demo manually, follow these steps:

1. **Create a New Directory:** Create a new directory within the `demonstrations_v2` folder, named after your demo. All demos are saved in the `demonstrations_v2` directory. For example: `demonstrations_v2/my_demo`.

2. **Add Required Files:** At a minimum, your demo directory should contain:
   - `demo.py`: The main executable Python script for your demo.
   - `metadata.json`: A JSON file containing metadata about your demo (see [Metadata Guidelines](#metadata-guidelines) for details).
   - `requirements.in` (optional): A file listing any additional dependencies required by your demo (see [Dependency Management](#dependency-management) for details).


## Tutorial Content Guidelines

While you are encouraged to be creative with your demo, please keep the following guidelines in mind:

- **License:** All contributions must be made under the Apache 2.0 license.
- **Title:** The title should be clear and concise. If based on a research paper, the title should be similar to the paper being implemented.
- **Summary:** All demos should include a 1-3 sentence summary below the title. This summary should clearly state the goal and outcome of the demo and link to any relevant papers or resources used.
- **Code Clarity:** Code should be clearly commented and explained, either through ReST-formatted comments or standard Python comments.
- **Reproducibility:** If your content involves random variables or outputs, a fixed seed should be set for reproducibility.
- **Originality:** All content must be original or free to reuse subject to license compatibility. For example, if you are implementing someone else's research, obtain permission before reproducing exact figures. Otherwise, avoid direct screenshots from papers and instead refer to figures in the paper within the text.
- **Code Review:** All submissions must pass code review before being merged into the repository.
- **Branching:** If your demo uses the latest stable release of PennyLane, submit your PR against the `master` branch. If it requires cutting-edge development versions of PennyLane or relevant plugins, submit your PR against the `dev` branch instead.
- **Privacy:** By submitting your demo, you consent to our [Privacy Policy](https://pennylane.ai/privacy/).


## General Guidelines

- **Format:** Demos are written as executable Python scripts.
- **Jupyter Notebook Conversion:** If you prefer writing your demo in a Jupyter notebook, you can convert it to the required executable Python format by following the [Notebook to Demo Converter](https://github.com/PennyLaneAI/demos/tree/master/notebook_converter).
- **Naming Convention:** All demo directories must be saved in the `demonstrations_v2` directory. For example: `demonstrations_v2/my_demo`.
- **Frameworks:** New demos should avoid using `autograd` or `TensorFlow`. `JAX` and `PyTorch` are recommended instead. Whenever possible, the use of `lightning.qubit` is also encouraged.
- **Restructured Text (ReST):** ReST sections can be included anywhere within the script by beginning the comment with 79 hash characters (`#`). These are useful for breaking up large code blocks and providing extensive explanations.
- **LaTeX Macros:** Avoid using LaTeX macros within your comments. Even if they appear to work in development, they will not be displayed correctly once the demo is published.
- **Author Information:** Include the author's information in the metadata.json file. See the metadata guidelines below.
- **Code Formatting:** Before submitting, run your script through the [Black Python formatter](https://github.com/psf/black):
    ```bash
    pip install black
    black -l 100 /demonstrations_v2/your_demo/demo.py
    ```
- **Referencing Other Demos in `demo.py`:** You can reference other demos in your `demo.py` file using ``:doc:\`demos/<demo_name>\``` syntax. This will create a link to the specified demo in the documentation. For example, to reference the `tutorial_qft` demo, use ``:doc:\`demos/tutorial_qft\``` .



## Image Guidelines

### File Size and Format

- Always optimize images for web use. Aim for a file in kilobytes (KB) rather than megabytes (MB).
- Use `.png` for all static images (decorative, descriptive, logos, etc.).
- Use `.gif` for animated images.

### Dimensions

- To maintain quality and performance, every image should be twice (2X) its visible dimension size on the web page, and at a minimum of `150 ppi/dpi` (preferably `300 ppi/dpi`).

## Metadata Guidelines

Every demo requires an accompanying `metadata.json` file located in its directory. This JSON file stores crucial information about the demo. Refer to [this example](https://github.com/PennyLaneAI/demos/blob/master/demonstrations_v2/tutorial_here_comes_the_sun/metadata.json) for the required fields:

- **Filename:** Ensure the file is named `metadata.json`.
- **Authors:** Ensure that the `"authors"` field is populated with the `"username"` of the author(s) as registered on pennylane.ai. This should match the username provided when creating the bio.
- **`executable_stable` and `executable_latest`:** Set either of these fields to `true` if your demo is executable with the stable or latest versions of PennyLane, respectively. If neither field is set to `true`, the demo will not be executable.
- **Dates:** Leave publication and modification dates empty if you do not know them or use UTC format (e.g., `"2023-10-01T12:00:00Z"`).
- **Categories:** Choose relevant categories for your demo, such as `"Getting Started"`.
- **`previewImages`:** Modify the final part of the image file names to match your demo's name. These two images will be provided to you once the review process begins. After receiving them, you must upload them to the address indicated in the metadata.
- **`relatedContent`:** This field refers to demos related to yours. You will need to provide the demo ID (e.g., `tutorial_qft`), weight (default is `1.0`) and the type as `demonstration`. This helps users find similar demos and enhances the discoverability of your content.
- **Questions:** If you have any doubts about a specific field, do not hesitate to post a comment for your demo's reviewer.

### Example `metadata.json` Structure

```json
{
    "title": "The hidden cut problem for locating unentanglement",
    "authors": [
        {
            "username": "simidzija"
        }
    ],
    "executable_stable": true,
    "executable_latest": true,
    "dateOfPublication": "2025-07-25T10:00:00+00:00",
    "dateOfLastModification": "2025-07-25T10:00:00+00:01",
    "categories": [
        "Algorithms"
    ],
    "tags": [],
    "previewImages": [
        {
            "type": "thumbnail",
            "uri": "/_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_hidden_cut.png"
        },
        {
            "type": "large_thumbnail",
            "uri": "/_static/demo_thumbnails/large_demo_thumbnails/thumbnail_large_hidden_cut.png"
        }
    ],
    "seoDescription": "Learn about a quantum algorithm that determines how to cut a many-body quantum state into unentangled components.",
    "doi": "",
    "references": [
        {
            "id": "Bouland2024",
            "type": "article",
            "title": "The State Hidden Subgroup Problem and an Efficient Algorithm for Locating Unentanglement",
            "authors": "Adam Bouland, Tudor Giurgică-Tiron, John Wright",
            "year": "2024",
            "journal": "STOC '25",
            "doi": "10.1145/3717823.3718118",
            "url": "https://doi.org/10.1145/3717823.3718118"
        }
    ],
    "basedOnPapers": [
        "10.1145/3717823.3718118"
    ],
    "referencedByPapers": [],
    "relatedContent": [
        {
            "type": "demonstration",
            "id": "tutorial_qft",
            "weight": 1.0
        },
        {
            "type": "demonstration",
            "id": "tutorial_qft_arithmetics",
            "weight": 1.0
        },
        {
            "type": "demonstration",
            "id": "tutorial_period_finding",
            "weight": 1.0
        }
    ]
}
```

### Validate Your Metadata File

```bash
pip install check-jsonschema 'jsonschema[format]'
check-jsonschema \
  --schemafile metadata_schemas/demo.metadata.schema.<largest_number>.json \
  demonstrations_v2/<your_demo_name>/metadata.json
```

Once your script and metadata are ready, you can submit a pull request!

## Dependency Management

Demo dependencies are automatically installed by the CLI tool during demo execution. For detailed information on dependency specifications, refer to [dependencies/README.md](/dependencies/README.md). A `requirements.txt` file will be created in the demo directory after a successful build.

## Building and Testing Locally

You can build and test PennyLane demos locally using the CLI tool, allowing you to preview your demo before submitting it for review.

> **Note:** Local HTML builds will differ visually from the production site or PR previews. These differences are expected; local builds are intended for functional testing rather than visual accuracy.

### Building a Demo in HTML Format

To build your demo in HTML format, navigate to the root directory of the repository and run:

```bash
demo build --format html <name_of_your_demo>
```

The generated HTML file will be located at:

```
_build/html/demos/<name_of_your_demo>.html
```

Open this file in your web browser to view the demo content.

### Building a Demo in JSON Format

To build your demo in JSON format, use:

```bash
demo build --format json <name_of_your_demo>
```

The output will be generated in the following directories:

- `_build/json/demos/<name_of_your_tutorial>/json` (contains various JSON and FJSON files)
- `_build/json/demos/<name_of_your_tutorial>/pack` (contains a Jupyter notebook `.ipynb`, a Python script `.py`, and other related files)

> **Note:** You may also add the `--execute` option to execute the demo during the build process.

If you do not specify a `--format` option, the demo will be built in JSON format by default.

## Support

- **Source Code:** https://github.com/PennyLaneAI/demos
- **Issue Tracker:** https://github.com/PennyLaneAI/demos/issues

If you encounter any issues, please report them on our GitHub issue tracker.

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please read and respect the [Code of Conduct](/.github/CODE_OF_CONDUCT.md).

## License

The materials and demos in this repository are **free** and **open source**, released under the Apache License, Version 2.0.

Please note, the file `custom_directives.py` is available under the BSD 3-Clause License, with copyright © 2017, PyTorch contributors.
