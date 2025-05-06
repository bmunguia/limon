import numpy as np

import pymeshb

# Reading a mesh with coordinates only
meshpath_in = "libMeshb/sample_meshes/quad.meshb"
coords, elements, solution = pymeshb.read_mesh(meshpath_in)
print(f"Available solution fields: {list(solution.keys())}")

# Write the mesh to a new file
meshpath_out = "output/quad.meshb"
pymeshb.write_mesh(meshpath_out, coords, elements,
                   solution=solution)

# Write the mesh with solution to a new file
meshpath_out = "output/quad_with_sol.meshb"
solpath_out = "output/quad_with_sol.solb"
solution = {
    "Temperature": coords[:, 0] + 10,
    "Velocity": np.zeros((coords.shape[0], coords.shape[1])),
}
solution["Velocity"][:, 1] = coords[:, 1] * 2
pymeshb.write_mesh(meshpath_out, coords, elements,
                   solpath=solpath_out, solution=solution)
