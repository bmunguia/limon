import libmeshb

filename = "triangles.meshb"
GmfRead = 1
GmfWrite = 2
version = 4
dim = 2

try:
    mesh_id = libmeshb.GmfOpenMesh(filename, GmfRead, version, dim)
    print(f"Mesh opened successfully with ID: {mesh_id}")
    print(f"Version: {version}")
    print(f"Dim: {dim}")
except Exception as e:
    print(f"Failed to open mesh: {e}")