"""Execute analysis.ipynb with the current Python environment, then validate."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
path = root / "notebooks" / "analysis.ipynb"
nb = nbformat.read(path, as_version=4)
for index, cell in enumerate(nb.cells):
    cell.setdefault("id", f"analysis-{index:03d}")
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(nb, km=km, timeout=180, resources={"metadata": {"path": str(path.parent)}})
client.execute()
nbformat.validate(nb)
errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
assert not errors, errors
nbformat.write(nb, path)
print(f"Executed {len(nb.cells)} cells without errors.")
