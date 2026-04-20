# HuggingFace Spaces entry point.
# HF Spaces runs: streamlit run app.py
#
# Instead of exec() (which breaks __file__ resolution), we use
# subprocess to launch ui/app.py as the real Streamlit app.
# This preserves all relative path resolution inside ui/app.py.

import subprocess
import sys
import os

if __name__ == "__main__":
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path,
                    "--server.headless", "true"], check=True)
else:
    # When Streamlit runs `app.py` directly (not as __main__),
    # redirect via importlib to run ui/app.py in-process with correct paths.
    import importlib.util
    _ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "app.py")
    _spec = importlib.util.spec_from_file_location("heist_ui_app", _ui_path)
    _mod = importlib.util.module_from_spec(_spec)
    # Patch __file__ so Path(__file__).parent resolves to ui/ correctly
    _mod.__file__ = _ui_path
    _spec.loader.exec_module(_mod)
