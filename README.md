# limon
The **LI**brary for **M**esh **O**perations and **N**umerics (limon) is a Python API to handle mesh/solution I/O for SU2 and GMF formats, basic mesh refinement operations, and various operations on metric tensor fields.

## Installation

> [!IMPORTANT]
> limon is currently only supported for Linux.

1. Initialize the submodules:

    ```
    git submodule update --init --recursive
    ```

2. Create the `conda` environment from `environment.yml`:

    ```
    conda env create -f environment.yml
    ```

3. Load the environment:

    ```
    conda activate meshb
    ```

4. From the top-level directory of the repo, install the package:

    ```
    pip install .
    ```

   Optional: to be able to run the tests, install with optional dependencies:

    ```
    pip install ".[test]"
    ```
