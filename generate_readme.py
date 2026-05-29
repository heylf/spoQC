# In[]
from pathlib import Path
docs_dir = "docs/source/"
parts = ["readme_header.md", "intro.md", "readme_doc_statement.md", "installation.md", "run.md", "contribute.md"]

content = "\n\n".join(
    Path(f"{docs_dir}/{p}")
    .read_text()
    .replace("_static", f"{docs_dir}/_static")
    .replace("```{note}", ">[!NOTE]")
    .replace("```", "")
    for p in parts
)

Path("README.md").write_text(content)