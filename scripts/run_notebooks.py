"""Execute every notebook in a fresh kernel using this environment's Python."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.chdir(root)
kernel_root = root / ".runtime" / "jupyter"
spec = kernel_root / "kernels" / "llamaindex-goodmem-rag"
spec.mkdir(parents=True, exist_ok=True)
(spec / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "LlamaIndex GoodMem RAG",
            "language": "python",
        }
    )
)
os.environ["JUPYTER_PATH"] = str(kernel_root) + os.pathsep + os.getenv("JUPYTER_PATH", "")
parser = argparse.ArgumentParser()
parser.add_argument("notebooks", nargs="*")
parser.add_argument("--output", default=".runtime/executed")
args = parser.parse_args()
output = root / args.output
output.mkdir(parents=True, exist_ok=True)
previous = output / "results.json"
results = json.loads(previous.read_text()) if previous.exists() else []
paths = [root / name for name in args.notebooks] or sorted(root.glob("*.ipynb"))
for path in paths:
    start = time.monotonic()
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="llamaindex-goodmem-rag",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbformat.write(notebook, output / path.name)
    results = [r for r in results if r["notebook"] != path.name]
    results.append(
        {"notebook": path.name, "passed": True, "seconds": round(time.monotonic() - start, 3)}
    )
    print(f"PASS {path.name} ({results[-1]['seconds']}s)", flush=True)
(output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
