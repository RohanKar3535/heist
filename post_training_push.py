"""
post_training_push.py — Run this after training finishes.

What it does:
  1. Generates all 6 chart PNGs from training JSONs
  2. Injects chart images into README.md (Training Results section)
  3. Pushes everything to GitHub main
  4. Pushes everything to HF Space

Usage (Kaggle cell):
    %env GITHUB_TOKEN=ghp_xxxx
    %env HF_TOKEN=hf_xxxx
    %env GITHUB_REPO=RohanKar3535/heist
    %env HF_SPACE=Rohan333555/heist-demo
    !cd /kaggle/working/heist && python post_training_push.py
"""

import os
import sys
import subprocess
import shutil
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Config from env vars ─────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
GITHUB_REPO  = os.environ.get("GITHUB_REPO", "RohanKar3535/heist")
HF_SPACE     = os.environ.get("HF_SPACE",    "Rohan333555/heist-demo")

def run(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT,
                            capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"  [STDERR] {result.stderr.strip()}")
        if check:
            raise RuntimeError(f"Command failed: {cmd}")
    return result


# ── Step 1: Generate charts ──────────────────────────────────────────────────
def step_generate_charts():
    print("\n📊 Step 1: Generating charts...")
    sys.path.insert(0, ROOT)
    from generate_charts import main as gen_charts
    paths = gen_charts()
    return paths


# ── Step 2: Inject charts into README ───────────────────────────────────────
def step_update_readme():
    print("\n📝 Step 2: Updating README with chart images...")
    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    charts_section = """
## 📊 Training Charts

### F1 & Reward Training Curve
![Training Curve](charts/01_training_curve.png)

### Red Queen ELO Leaderboard
![ELO Leaderboard](charts/02_elo_leaderboard.png)

### Investigator Weakness Heatmap
![Weakness Heatmap](charts/03_weakness_heatmap.png)

### Baseline Comparison
![Baseline Comparison](charts/04_baseline_comparison.png)

### Episode Distribution by Scheme Type
![Scheme Distribution](charts/05_scheme_distribution.png)

### Criminal vs Investigator ELO Arms Race
![ELO Arms Race](charts/06_criminal_elo_race.png)

"""

    # Insert before the "## 🛠️ Setup" section
    marker = "## 🛠️ Setup"
    if marker in content:
        # Remove any existing charts section first
        if "## 📊 Training Charts" in content:
            start = content.index("## 📊 Training Charts")
            end   = content.index(marker)
            content = content[:start] + content[end:]
        content = content.replace(marker, charts_section + marker)
    else:
        # Append at end
        if "## 📊 Training Charts" not in content:
            content += charts_section

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  ✅ README.md updated with chart images")


# ── Step 3: Push to GitHub ───────────────────────────────────────────────────
def step_push_github():
    print("\n🐙 Step 3: Pushing to GitHub...")
    if not GITHUB_TOKEN:
        print("  [SKIP] GITHUB_TOKEN not set")
        return

    remote = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
    run(f'git config user.email "heist-bot@hackathon.ai"')
    run(f'git config user.name "HEIST Bot"')
    run(f'git remote set-url origin "{remote}"')

    # Stage everything important
    run('git add README.md charts/ training_curves.json weakness_history.json '
        'benchmark_results.json zero_day_scheme.json zero_day_visualization.json '
        'training_log.csv f1_history.json 2>/dev/null || true', check=False)

    # Check if there's anything to commit
    status = run('git status --porcelain', check=False)
    if not status.stdout.strip():
        print("  [SKIP] Nothing new to commit for GitHub")
        return

    run('git commit -m "Add training results, charts, and updated README"')
    run('git push origin main')
    print("  ✅ Pushed to GitHub")


# ── Step 4: Push to HF Space via huggingface_hub ─────────────────────────────
def step_push_hf_space():
    print("\n🤗 Step 4: Pushing to HF Space...")
    if not HF_TOKEN:
        print("  [SKIP] HF_TOKEN not set")
        return

    try:
        from huggingface_hub import HfApi, upload_file, upload_folder
    except ImportError:
        run("pip install huggingface_hub -q", check=False)
        from huggingface_hub import HfApi, upload_file, upload_folder

    api = HfApi(token=HF_TOKEN)
    repo_id = HF_SPACE
    repo_type = "space"

    # Files to upload (text/json — via git)
    text_files = [
        "README.md",
        "training_curves.json",
        "weakness_history.json",
        "benchmark_results.json",
        "zero_day_scheme.json",
        "zero_day_visualization.json",
        "training_log.csv",
        "f1_history.json",
    ]

    HF_YAML = (
        "---\n"
        "title: HEIST\n"
        "emoji: 🕵️\n"
        "colorFrom: red\n"
        "colorTo: purple\n"
        "sdk: streamlit\n"
        'sdk_version: "1.32.0"\n'
        "app_file: src/streamlit_app.py\n"
        "pinned: true\n"
        "---\n\n"
    )

    for fname in text_files:
        src = os.path.join(ROOT, fname)
        if not os.path.exists(src):
            continue
        try:
            if fname == "README.md":
                # HF Space README must have YAML front matter
                with open(src, encoding="utf-8") as f:
                    body = f.read()
                if not body.startswith("---"):
                    body = HF_YAML + body
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                                 delete=False, encoding="utf-8") as tmp:
                    tmp.write(body)
                    upload_src = tmp.name
                api.upload_file(path_or_fileobj=upload_src, path_in_repo=fname,
                                repo_id=repo_id, repo_type=repo_type,
                                commit_message="Update README.md")
                os.unlink(upload_src)
            else:
                api.upload_file(path_or_fileobj=src, path_in_repo=fname,
                                repo_id=repo_id, repo_type=repo_type,
                                commit_message=f"Update {fname}")
            print(f"  ✅ {fname}")
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")

    # Upload charts/ folder (PNGs — uses Xet storage automatically)
    charts_src = os.path.join(ROOT, "charts")
    if os.path.exists(charts_src):
        try:
            api.upload_folder(
                folder_path=charts_src,
                path_in_repo="charts",
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message="Add training charts",
            )
            print("  ✅ charts/ (all PNGs)")
        except Exception as e:
            print(f"  [WARN] charts/: {e}")

    # Upload updated Streamlit app
    ui_src = os.path.join(ROOT, "ui", "app.py")
    if os.path.exists(ui_src):
        try:
            api.upload_file(
                path_or_fileobj=ui_src,
                path_in_repo="src/streamlit_app.py",
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message="Update Streamlit app",
            )
            print("  ✅ src/streamlit_app.py")
        except Exception as e:
            print(f"  [WARN] streamlit_app.py: {e}")

    print("  ✅ HF Space updated")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "="*60)
    print("🏁  POST-TRAINING PUSH COMPLETE")
    print("="*60)

    tc_path = os.path.join(ROOT, "training_curves.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        eps  = tc["episodes"]
        f1s  = tc["f1"]
        rvs  = tc["r_inv"]
        print(f"  Episodes      : {len(eps)}")
        print(f"  Best F1       : {max(f1s):.3f}")
        print(f"  Mean F1       : {sum(f1s)/len(f1s):.3f}")
        print(f"  R_inv range   : {min(rvs):.3f} – {max(rvs):.3f}")

    charts_dir = os.path.join(ROOT, "charts")
    if os.path.exists(charts_dir):
        chart_files = [f for f in os.listdir(charts_dir) if f.endswith(".png")]
        print(f"  Charts        : {len(chart_files)} PNG files")

    print(f"\n  🐙 GitHub : https://github.com/{GITHUB_REPO}")
    print(f"  🤗 Space  : https://huggingface.co/spaces/{HF_SPACE}")
    print("="*60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 HEIST Post-Training Push")
    print(f"   GitHub : {GITHUB_REPO}")
    print(f"   Space  : {HF_SPACE}")

    step_generate_charts()
    step_update_readme()
    step_push_github()
    step_push_hf_space()
    print_summary()
