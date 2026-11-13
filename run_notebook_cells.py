import os
import sys
import json
import base64
import io
from pathlib import Path

# Setup paths
job_root = Path(r"c:\Desktop\Table\Job")
task1_dir = job_root / "task_1_video_classifier"
nb_path = task1_dir / "notebooks" / "walkthrough.ipynb"

# Ensure task 1 root and mplconfig are configured
os.environ["MPLCONFIGDIR"] = str(task1_dir / "data" / ".mplconfig")
sys.path.insert(0, str(task1_dir))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global namespace for notebook
shared_ns = {
    "__file__": str(nb_path),
    "__name__": "__main__",
}

print(f"Executing {len(nb['cells'])} cells in Task 1 notebook...")

exec_counter = 1
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue

    code = "".join(cell["source"])
    print(f"\n--- Running Code Cell {exec_counter} (Cell index {idx}) ---")
    
    # Intercept stdout and plots
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf
    
    # Track figures
    plt.close('all')
    
    error_occurred = None
    try:
        # If last line is an expression, evaluate it like jupyter does
        lines = code.strip().split("\n")
        # Pre-process lines to avoid plt.show() blocking
        exec(code, shared_ns)
    except Exception as e:
        error_occurred = e
    finally:
        sys.stdout = old_stdout

    if error_occurred:
        print(f"ERROR in cell {exec_counter}: {error_occurred}")
        import traceback
        traceback.print_exc()
        raise error_occurred

    captured_out = stdout_buf.getvalue()
    cell["outputs"] = []
    cell["execution_count"] = exec_counter
    
    # If there was text output
    if captured_out:
        print(f"Captured stdout ({len(captured_out)} chars):")
        for line in captured_out.strip().split("\n")[:4]:
            print(f"  {line}")
        cell["outputs"].append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in captured_out.splitlines()]
        })

    # If there are open matplotlib figures
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
        img_buf.seek(0)
        b64_png = base64.b64encode(img_buf.read()).decode("ascii")
        cell["outputs"].append({
            "data": {
                "image/png": b64_png,
                "text/plain": [f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"]
            },
            "metadata": {},
            "output_type": "display_data"
        })
        print(f"  Captured figure {fig_num} (PNG: {len(b64_png)} chars)")
        plt.close(fig)

    exec_counter += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("\nSUCCESS: All cells in Task 1 walkthrough.ipynb executed and outputs embedded!")
