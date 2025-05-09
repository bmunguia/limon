# pymeshb
pymeshb is a Python API to handle the *.meshb file format, and to handle various operations on metric tensor fields.

## Installation

**Note:** pymeshb is currently only supported for Linux.

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
    poetry install
    ```
