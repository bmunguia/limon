import numpy as np

import pymeshb

# Reading a mesh with coordinates only
input_filepath = "libMeshb/sample_meshes/quad.meshb"
coords, solution = pymeshb.read_mesh(input_filepath, read_sol=True)
print(f"Available solution fields: {list(solution.keys())}")

# Write the mesh to a new file
output_filepath = "output/quad.meshb"
pymeshb.write_mesh(output_filepath, coords, solution)

# Write the mesh with solution to a new file
output_filepath = "output/quad_with_sol.meshb"
solution = {
    "Temperature": np.ones(coords.shape[0]),
    "Velocity": np.zeros((coords.shape[0], coords.shape[1])),
}
solution["Velocity"][:, 1] = 2
pymeshb.write_mesh(output_filepath, coords, solution)
