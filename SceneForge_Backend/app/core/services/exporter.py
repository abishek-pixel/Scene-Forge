def export_mesh(points: list) -> str:
    # Dummy export function; save points to a file in OBJ format for example
    filename = "output_mesh.obj"
    with open(filename, "w") as f:
        f.write("# OBJ file\n")
        for p in points:
            f.write(f"v {p} {p} {p}\n")  # simplistic: all coords same for demo
    return filename