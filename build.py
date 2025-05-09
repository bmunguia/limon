import os
import subprocess
import shutil
from pathlib import Path

def build(setup_kwargs=None):
    """Build the C++ extensions using CMake"""
    # Get conda environment path
    conda_prefix = os.environ.get("CONDA_PREFIX")
    
    # Create build directory
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Run CMake
    cmd = [
        "cmake", 
        "-S", ".", 
        "-B", "build",
        "-DCMAKE_BUILD_TYPE=Release"
    ]
    if conda_prefix:
        cmd.extend(["-DCMAKE_PREFIX_PATH=" + conda_prefix])
    
    subprocess.run(cmd, check=True)
    
    # Build
    subprocess.run(["cmake", "--build", "build"], check=True)
    
    # Create necessary directory structure
    metrics_dir = Path("pymeshb/metric")
    metrics_dir.mkdir(exist_ok=True)
    
    # Copy built libraries to right locations
    metric_lib = list(build_dir.glob("*_metric.*"))
    gamma_lib = list(build_dir.glob("*libmeshb.*"))
    
    if metric_lib:
        shutil.copy2(metric_lib[0], metrics_dir)
    else:
        print("Warning: _metric library not found in build dir")
        
    if gamma_lib:
        shutil.copy2(gamma_lib[0], Path("pymeshb/gamma"))
    else:
        print("Warning: libmeshb library not found in build dir")
        
    # Return empty dict if called by Poetry
    return {} if setup_kwargs is not None else None

if __name__ == "__main__":
    build()