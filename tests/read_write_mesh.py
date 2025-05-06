import pymeshb

input_filepath = "libMeshb/sample_meshes/quad.meshb"
nodes, coords = pymeshb.read_mesh(input_filepath)

if nodes is not None:
    print("\nFirst 5 nodes:")
    for i in range(min(5, len(nodes))):
        print(f"Node {nodes[i]}: {coords[i]}")

    # Write the mesh to a new file
    output_filepath = "output/quad_output.meshb"
    success = pymeshb.write_mesh(nodes, coords, output_filepath)

    if success:
        # Verify by reading back
        print("\nVerifying output:")
        nodes2, coords2 = pymeshb.read_mesh(output_filepath)

        if nodes2 is not None:
            print("\nComparison of first 5 nodes:")
            for i in range(min(5, len(nodes))):
                print(f"Original: Node {nodes[i]}: {coords[i]}")
                print(f"Output:   Node {nodes2[i]}: {coords2[i]}")