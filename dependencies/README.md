# Dependency Specification

This directory houses the files that specify and constrain the dependencies required for PennyLane demonstrations. These specifications ensure consistent and reproducible build and execution environments for all demos.

## Table of Contents

*   [Overview](#overview)
*   [Dependency Files](#dependency-files)
    *   [`constraints-dev.txt`](#constraints-devtxt)
    *   [`constraints-plc-dev.json`](#constraints-plc-devjson)
    *   [`constraints-stable.txt`](#constraints-stabletxt)
    *   [`requirements-build.txt`](#requirements-buildtxt)
    *   [`requirements-core.in`](#requirements-corein)

---

## Overview

Managing dependencies is crucial for the reliability and reproducibility of our PennyLane demonstrations. This directory contains dedicated files that define the precise versions of packages used in different contexts—ranging from development builds to stable releases and core execution requirements.

## Dependency Files

The following files define the various dependency sets:

### `constraints-dev.txt`

**NOTE**: The development versions of Lightning, Catalyst, and PennyLane require a specific installation order, and as such their versions are not controlled by this file. See the next section for their installation.

This file specifies the allowed versions of dependencies for building and running demonstrations when targeting the most recent **development builds** of PennyLane and its associated plugins. The actual dependencies are defined in the requirements files; the constraints file only restricts which versions can be installed. This ensures compatibility with cutting-edge features and ongoing development. Unless a new global development dependency is being added, you likely don't need to modify this file.

*   **Location:** `./constraints-dev.txt`
*   **Purpose:** Defines dependencies for the `demo build --dev` command.

### `constraints-plc-dev.json`

This file specifies the allowed release versions of PennyLane, Lightning, and Catalyst for **development builds**. These packages are currently installed **only if the demo is set to be executed** (`demo build --execute --dev`). The latest version available on test-pypi, whose release version matches the constraint, is what gets installed (e.g. 0.45.0.dev15 == 0.45.0, 0.45.0rc7 == 0.45.0, 0.46.0.dev3 != 0.45.0, etc.). Only update this constraint after a new release, so that the release candidate versions continue to be used during feature freeze. Unless you're a release manager, you likely don't need to modify this file.

*   **Location:** `./constraints-plc-dev.json`
*   **Purpose:** Defines PennyLane, Lightning, and Catalyst versions for the `demo build --execute --dev` command.
*   **Format:** JSON array of package objects, with the package name as it appears on PyPi, and the major version of the upcoming release. Order of entries must be preserved (Catalyst, then Lightning, then PennyLane).

### `constraints-stable.txt`

This file specifies the exact dependency versions used for building and running demonstrations against the most recent **stable release** of PennyLane and its associated plugins. This ensures compatibility and stability for production-ready demonstrations. If your pull request is targeting the `master` branch, then these are the dependency constraints that will be used. 

*   **Location:** `./constraints-stable.txt`
*   **Purpose:** Defines dependencies for the `demo build --no-dev` (default) command.

### `requirements-build.txt`

This file lists the dependencies specifically required for the build process of the demo notebooks using Sphinx. These packages are essential for converting and rendering the Python scripts into web-viewable formats (HTML/JSON). If you are simply creating or updating a demo, it is very unlikely that you will need to modify this file.

*   **Location:** `./requirements-build.txt`
*   **Purpose:** Specifies tools needed for the Sphinx-based build system.

### `requirements-core.in`

This file specifies the fundamental execution dependencies that are automatically installed and available to **all** QML demonstrations by default. If you are contributing a new demo, it is unlikely that you will need to modify this file. Additional dependencies, unique to a specific demo, can be specified in the individual `requirements.in` files within the respective demo directories.

*   **Location:** `./requirements-core.in`
*   **Purpose:** Defines baseline dependencies for all demos.
