# build.py
import subprocess, sys, os

if __name__ == "__main__":
    if not shutil.which("pyinstaller"):
        print("PyInstaller not found. Install: pip install pyinstaller")
        sys.exit(1)
    # simple pyinstaller invocation
    cmd = [
        "pyinstaller",
        "--onefile",
        "--add-data", "templates;templates",
        "--add-data", "static;static",
        "app.py"
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Done. Check the dist/ folder.")
