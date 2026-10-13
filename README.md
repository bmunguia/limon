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
    conda activate limon
    ```

4. From the top-level directory of the repo, install the package:

    ```
    pip install .
    ```

   Optional: to be able to run the tests, install with optional dependencies:

    ```
    pip install ".[test]"
    ```

## Usage

limon reads and writes meshes and solutions as plain Python dictionaries of NumPy arrays. The format follows the file extension: meshes in SU2 (`.su2`) or GMF (`.mesh`, `.meshb`), solutions as SU2 restarts (`.dat` binary, `.csv` ASCII) or GMF (`.sol`, `.solb`). Every I/O function raises `limon.mesh.LimonIOError` when a file cannot be read or written.

### Mesh

`load_mesh` and `write_mesh` handle a mesh on its own:

```python
from limon.mesh import load_mesh, write_mesh

# SU2 meshes carry their marker names
mesh = load_mesh('NACA0012_inv.su2')
mesh['dim'], mesh['num_point']        # (2, 5233)
mesh['coords']                        # (5233, 2) float64
mesh['elements']['Triangles']         # (10216, 4) uint32, rows [node0, node1, node2, ref], e.g. [417, 69, 311, 0]
mesh['boundaries']['Edges']           # (250, 3) uint32, rows [node0, node1, marker ID], e.g. [199, 0, 1]
mesh['markers']                       # {1: 'airfoil', 2: 'farfield'}

# Convert to GMF: writes naca.meshb plus naca.names.json holding the marker names
write_mesh('naca.meshb', mesh)
load_mesh('naca.meshb')['markers']    # {1: 'airfoil', 2: 'farfield'} (from naca.names.json)

# A GMF mesh from another tool (no sidecar): its boundary elements carry only marker IDs
load_mesh('adapted.meshb')['markers'] # {}
adapted = load_mesh('adapted.meshb', marker_map={1: 'airfoil', 2: 'farfield'})
adapted['markers']                    # {1: 'airfoil', 2: 'farfield'}
write_mesh('adapted.su2', adapted)    # MARKER_TAG= airfoil, MARKER_TAG= farfield
```

GMF files store only an integer reference per boundary element, not a marker name. limon keeps the names in a small JSON sidecar, `<stem>.names.json`, next to the GMF file:
- `write_mesh` to `.mesh`/`.meshb` writes the sidecar from `mesh['markers']` (or from `marker_map=` if given). `write_names=False` skips it.
- `load_mesh` of a GMF file takes the names from `marker_map=` if given, else from the sidecar if present, else leaves `markers` empty. The boundary elements still carry their integer references, and writing such a mesh to `.su2` names the markers `REF_<n>`.
- SU2 meshes always carry their names; `load_mesh` ignores `marker_map` for them.

The mesh dictionary contains:

| Key | Contents |
|---|---|
| `coords` | `float64` array `(num_point, dim)` of node coordinates |
| `elements` | Interior elements by type (`Triangles`, `Quadrilaterals`, `Tetrahedra`, `Prisms`, `Pyramids`, `Hexahedra`): `uint32` arrays `(num_elem, num_node + 1)` of 0-based node indices, the last column being the element reference |
| `boundaries` | Boundary elements by type (`Edges` in 2-D, `Triangles`/`Quadrilaterals` in 3-D) in the same layout, with the marker ID in the last column; `Corners` holds the node indices of corner points when the file has them |
| `markers` | Marker ID → marker name |
| `dim`, `num_point` | Mesh dimension and number of points |

For example, the airfoil's boundary edges are `edges[edges[:, -1] == 1, :2]` with `edges = mesh['boundaries']['Edges']`. On write, interior and boundary elements are oriented the way SU2 checks for them (all `.su2` output and 2-D `.mesh`/`.meshb`), so SU2 finds nothing to re-orient.

### Mesh and solution

`load_mesh_and_solution` and `write_mesh_and_solution` handle both files at once; `load_solution` and `write_solution` handle a solution on a mesh already in memory:

```python
from limon.mesh import load_mesh_and_solution, load_solution, write_mesh_and_solution, write_solution

# Mesh plus solution in one dictionary
data = load_mesh_and_solution('NACA0012_inv.su2', 'restart_flow.dat')
data['markers']                       # {1: 'airfoil', 2: 'farfield'}, as from load_mesh
data['labels']                        # {1: 'x', 2: 'y', 3: 'Density', 4: 'Momentum', 5: 'Energy'}
data['solution']['Density']           # (num_point,)
data['solution']['Momentum']          # (num_point, 2)
# With SENSOR_HESSIAN and METRIC in the run's VOLUME_OUTPUT, the restart also holds:
# data['solution']['Hessian_Mach']    # (num_point, 3), packed [xx, xy, yy]
# data['solution']['Metric']          # (num_point, 3), packed [xx, xy, yy]

# Convert both to GMF: naca.meshb, naca.solb and one naca.names.json with markers and field names
write_mesh_and_solution('naca.meshb', 'naca.solb', data)

# Solution only: needs the point count and dimension of its mesh
n, dim = data['num_point'], data['dim']
density = load_solution('restart_flow.dat', n, dim)['solution']['Density']
write_solution('density.solb', {'Density': density}, n, dim)

# A GMF solution from another tool (no sidecar): fields are unnamed unless named in file order
load_solution('adapted.solb', n, dim)['labels']                    # {1: 'REF_1'}
load_solution('adapted.solb', n, dim, names=['Metric'])['labels']  # {1: 'Metric'}
metric = load_solution('adapted.solb', n, dim, names=['Metric'])['solution']['Metric']
```

GMF solutions store no field names either. `write_solution` records them (in dictionary order) in the same `<stem>.names.json` sidecar, so a mesh and solution sharing a stem share one sidecar. `load_solution` takes the names from `names=` if given, else from the sidecar, else calls the fields `REF_1`, `REF_2`, … `load_mesh_and_solution` accepts both `marker_map=` and `names=`.

The solution dictionary contains:

| Key | Contents |
|---|---|
| `solution` | Field name → `float64` array: scalars `(num_point,)`, vectors `(num_point, dim)`, symmetric tensors `(num_point, 3)` in 2-D and `(num_point, 6)` in 3-D |
| `labels` | 1-based field index → field name, in file order |

Symmetric tensors (Hessians, metrics) are packed as `[xx, xy, yy]` in 2-D and `[xx, xy, yy, xz, yz, zz]` in 3-D. SU2 restart columns are grouped by name: `Momentum_x`, `Momentum_y` become `Momentum`, and `Metric_xx`, `Metric_xy`, `Metric_yy` become `Metric`.

### Surface metric

`limon.geometry.surface_metric` builds the surface-geometry metric of a mesh adapter's background mesh (2-D only; 3-D raises `NotImplementedError`):

```python
from limon.geometry import surface_metric
from limon.mesh import load_mesh, write_mesh_and_solution

mesh = load_mesh('cylinder.su2')
metric = surface_metric(mesh, {'cylinder': 2.0, 'farfield': 10.0}, mode='angle', hmin=1e-6, hmax=25.0)
mesh['solution'] = {'Metric': metric}
write_mesh_and_solution('back.meshb', 'back.solb', mesh)
```

Boundary nodes of the listed markers get a tangent edge length set by the local curvature and the deviation (`'angle'`: degrees of normal turn per edge; `'hausdorff'`: largest distance between edge and boundary), and the normal size `hmax`. All other nodes get the isotropic size `hmax`. A node on several listed markers uses the first one in the mapping. The result is a packed `(num_point, 3)` metric with every eigenvalue in `[1/hmax^2, 1/hmin^2]`.

### Work in progress

These modules are still being developed; their interfaces may change:
- `limon.numerics`: operations on fields of symmetric (metric) tensors, such as eigendecomposition and recomposition (`decompose_metric_field`, `recompose_metric_field`, `rotation_angles_field`), perturbation of eigenvalues and eigenvectors (`perturb_metric_field`), normalization to a target complexity (`integrate_metric_field`, `normalize_metric_field`), and edge lengths in the metric (`metric_edge_length`).
- `limon.operations.refine_mesh`: uniform mesh refinement.
