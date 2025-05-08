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
    "Metric": np.zeros((num_point, (num_dim * (num_dim + 1)) // 2))
}

# Make sample metric diagonal
# We store the lower triangle of the tensor
solution["Metric"][:, 0] = 1e1
solution["Metric"][:, 2] = 1e2
solution["Metric"][:, 5] = 1e3

# Write the mesh with solution to a new file
meshpath_out = "output/quad_with_met.meshb"
solpath_out = "output/quad_with_met.solb"

pymeshb.write_mesh(meshpath_out, coords, elements,
                   solpath=solpath_out, solution=solution)

# Test perturbations
print("Perturbing metric field")

# Create perturbation arrays for eigenvalues (in log space)
# For 3D metrics, we need perturbations of shape (num_points, 3)
val_perturbations = np.zeros((num_point, 3))
# Add a constant perturbation to the eigenvalues
val_perturbations[:, 0] = -4.6052
val_perturbations[:, 1] = 2.3025
val_perturbations[:, 2] = 2.3025
# # Add random perturbations between -0.1 and 0.1 to the log of eigenvalues
# val_perturbations[:, 0] = np.random.uniform(-0.1, 0.1, num_point)  # perturb first eigenvalue
# val_perturbations[:, 1] = np.random.uniform(-0.1, 0.1, num_point)  # perturb second eigenvalue
# val_perturbations[:, 2] = np.random.uniform(-0.1, 0.1, num_point)  # perturb third eigenvalue

# Create perturbation arrays for eigenvectors
vec_perturbations = np.zeros((num_point, num_dim, num_dim))

# Perturb the metric field
perturbed_metrics = pymeshb.metric.perturb_metric_field(
    solution["Metric"],
    val_perturbations,
    vec_perturbations
)

# Create a new solution dictionary for perturbed metrics
perturbed_solution = {
    "Metric": perturbed_metrics
}

# Write the mesh with perturbed metrics to new files
pert_meshpath_out = "output/quad_with_pert_met.meshb"
pert_solpath_out = "output/quad_with_pert_met.solb"

pymeshb.write_mesh(pert_meshpath_out, coords, elements,
                   solpath=pert_solpath_out, solution=perturbed_solution)