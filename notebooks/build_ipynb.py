"""Generate the uploadable Kaggle notebook from kaggle.py.

kaggle.py stays the source of truth -- it is readable in the repo and its
imports can be checked. This turns it into a .ipynb you upload straight to
Kaggle: `# %%` sections become cells, the section titles become headings, and
`# !shell` lines become real `!shell` lines that actually run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SOURCE = Path(__file__).with_name("kaggle.py")
TARGET = Path(__file__).with_name("remy-kaggle.ipynb")

HEADINGS = {
    "1. the repo": "### 1. Get the code\nTakes a minute. Needs **Internet: On** in the right-hand panel.",
    "2. the anchor": "### 2. Find your anchor\nFinds the image from the dataset you attached with **+ Add Input**.",
    "3. pick the anchor strength": (
        "### 3. Sweep — then stop and look\n"
        "Four versions of one prompt at different anchor strengths. **Look at all four.** "
        "Pick the one where she's recognisably herself but not copying the anchor's pose, "
        "and put that number in the next cell. This decides whether the big batch is worth anything."
    ),
    "4. the batch": "### 4. The batch\n45–90 minutes for 200. Resumes if the session drops — just re-run it.",
    "5. curate": "### 5. Curate\nScores every image against your anchor and keeps the best 40.",
    "6. train": "### 6. Train the LoRA\nThe long one. This is what makes her repeatable.",
    "7. take it home": "### 7. Download\nGrab the zip from the **Output** panel on the right.",
}


def cells_from(text: str) -> list[dict]:
    chunks = re.split(r"^# %% (.+?) -*\n", text, flags=re.M)
    doc, rest = chunks[0], chunks[1:]

    cells = [{
        "cell_type": "markdown", "metadata": {},
        "source": [
            "# Remy — image pipeline\n", "\n",
            "Run these top to bottom. Before you start, in the right-hand panel set\n",
            "**Accelerator** to a GPU and **Internet** to On, and attach your anchor\n",
            "image with **+ Add Input**.\n", "\n",
            "> " + doc.strip().splitlines()[0].strip('"'), "\n",
        ],
    }]

    for title, body in zip(rest[::2], rest[1::2]):
        heading = HEADINGS.get(title.strip())
        if heading:
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": [line + "\n" for line in heading.split("\n")]})
        # `# !pip ...` is a shell line commented out so the file stays importable.
        code = re.sub(r"^# (!.*)$", r"\1", body.strip("\n"), flags=re.M)
        cells.append({
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in code.split("\n")][:-1]
                      + [code.split("\n")[-1]],
        })
    return cells


def build() -> Path:
    notebook = {
        "cells": cells_from(SOURCE.read_text(encoding="utf-8")),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return TARGET


if __name__ == "__main__":
    path = build()
    data = json.loads(path.read_text())
    code = sum(1 for c in data["cells"] if c["cell_type"] == "code")
    print(f"{path}: {len(data['cells'])} cells ({code} code)")
