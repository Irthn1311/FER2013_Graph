import json
from pathlib import Path

nb_path = Path("notebooks/kaggle-end-to-end.ipynb")
content = nb_path.read_text(encoding="utf-8")

old_str = '''        "STAGE_OUTPUT_DIRS = {\\n",
        "    \\"stage1\\": OUTPUT_BASE / \\"d10_p5_stage1_supcon\\",\\n",
        "    \\"stage2\\": OUTPUT_BASE / \\"d10_p5_stage2_relation\\",\\n",
        "}\\n",
        "STAGE_LABELS = {\\n",
        "    \\"stage1\\": \\"D10-P5 Stage 1 SupCon\\",\\n",
        "    \\"stage2\\": \\"D10-P5 Stage 2 Relation\\",\\n",
        "}\\n",'''

new_str = '''        "STAGE_OUTPUT_DIRS = {\\n",
        "    \\"stage1\\": OUTPUT_BASE / Path(STAGE1_CONFIG_PATH).stem,\\n",
        "    \\"stage2\\": OUTPUT_BASE / Path(STAGE2_CONFIG_PATH).stem,\\n",
        "}\\n",
        "STAGE_LABELS = {\\n",
        "    \\"stage1\\": f\\"Stage 1 {Path(STAGE1_CONFIG_PATH).stem}\\",\\n",
        "    \\"stage2\\": f\\"Stage 2 {Path(STAGE2_CONFIG_PATH).stem}\\",\\n",
        "}\\n",'''

if old_str in content:
    content = content.replace(old_str, new_str)
    nb_path.write_text(content, encoding="utf-8")
    print("Successfully updated notebook output directories.")
else:
    print("Could not find the target string in notebook.")
