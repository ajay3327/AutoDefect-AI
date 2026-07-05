"""
Launcher script for AutoDefect AI.
"""

import os
import sys
import subprocess

def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    main_file = os.path.join(app_dir, "app", "main.py")

    if not os.path.exists(main_file):
        print(f"Error: Could not find {main_file}")
        sys.exit(1)

    print("Starting Streamlit Dashboard...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", main_file], cwd=app_dir)
    except KeyboardInterrupt:
        print("Server stopped.")

if __name__ == "__main__":
    main()
