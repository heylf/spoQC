# In[]
import re
from pathlib import Path
docs_dir = "docs/source/"
parts = ["readme_header.md", "intro.md", "readme_doc_statement.md", "installation.md", "run.md", "nextflow.md", "contribute.md"]

def transform(text: str) -> str:
    # Convert ```{note} ... ``` blocks to GitHub admonitions
    text = re.sub(
        r"```{note}\n(.*?)\n```",
        r"> [!NOTE]\n\1", # --> group 1 is fully taken over
        text,
        flags=re.DOTALL, # --> matches everything even when a newline is in the text (i.e., \n). Else it would stop.
    )

    text = re.sub(
        r"\[Run\]\(run\.md\)",
        r"[Run](#run)",
        text,
        flags=re.DOTALL,
    )

    return text.replace("_static", f"{docs_dir}/_static")

if __name__ == "__main__":
    content = "\n\n".join(
        transform(Path(f"{docs_dir}/{p}").read_text())
        for p in parts
    )
    Path("README.md").write_text(content)