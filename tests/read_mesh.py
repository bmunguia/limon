import libmeshb

filename = "triangles.meshb"
GmfRead = 1
GmfWrite = 2
version = 4  # Meshb version
dim = 2

try:
    mesh_id = libmeshb.GmfOpenMesh(filename, GmfRead, version, dim)
    print(f"Mesh opened successfully with ID: {mesh_id}")
except Exception as e:
    print(f"Failed to open mesh: {e}")