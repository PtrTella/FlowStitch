import nbformat

def list_structure(path):
    print(f"\n--- Structure of {path} ---")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            lines = cell.source.split("\n")
            headers = [line for line in lines if line.startswith("#")]
            if headers:
                print(f"Cell {i} (MD): {headers[0]}")
        elif cell.cell_type == "code":
            if "class " in cell.source:
                classes = [line for line in cell.source.split("\n") if line.startswith("class ")]
                print(f"Cell {i} (CODE - CLASSES): {classes}")
            else:
                first_line = cell.source.split("\n")[0] if cell.source else ""
                print(f"Cell {i} (CODE): {first_line[:50]}")

list_structure("notebooks/05_Decoder_Exploration.ipynb")
list_structure("notebooks/06_Spectral_Matting_and_TDA.ipynb")
