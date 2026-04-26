"""Push latest training results to HF Space."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN", ""))
REPO = "Rohan333555/heist-demo"

# Upload charts folder
api.upload_folder(
    folder_path="charts",
    path_in_repo="charts",
    repo_id=REPO,
    repo_type="space",
    commit_message="Update charts with new training results",
)
print("charts done")

# Upload data files
files = [
    "training_curves.json",
    "weakness_history.json",
    "benchmark_results.json",
    "zero_day_scheme.json",
    "zero_day_visualization.json",
    "training_log.csv",
    "f1_history.json",
]
for fname in files:
    if os.path.exists(fname):
        api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=fname,
            repo_id=REPO,
            repo_type="space",
            commit_message=f"Update {fname}",
        )
        print(f"{fname} done")

print("All done - HF Space updated!")
