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

# Create a sample solution dict
num_point = coords.shape[0]
num_dim = coords.shape[1]
solution = {
    "Temperature": coords[:, 0] + 10,
    "Velocity": np.zeros((num_point, num_dim)),
    "Metric": np.zeros((num_point, (num_dim * (num_dim + 1)) // 2))
}
solution["Velocity"][:, 1] = coords[:, 1] * 2

# Make sample metric diagonal
# We store the lower triangle of the tensor
solution["Metric"][:, 0] = 1e1
solution["Metric"][:, 2] = 1e2
solution["Metric"][:, 5] = 1e3

# Write the mesh with solution to a new file
meshpath_out = "output/quad_with_sol.meshb"
solpath_out = "output/quad_with_sol.solb"

pymeshb.write_mesh(meshpath_out, coords, elements,
                   solpath=solpath_out, solution=solution)
