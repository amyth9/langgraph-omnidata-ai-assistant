#!/usr/bin/env python3

import subprocess
import sys
import os


def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_dir = os.path.join(current_dir, "ui")

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8501",
            "--server.address",
            "localhost",
        ]

        print("🚀 Starting AI Assistant...")
        print(f"📁 Working directory: {ui_dir}")
        print("🌐 App will be available at: http://localhost:8501")
        print("⏹️ Press Ctrl+C to stop the app")
        print()

        subprocess.run(cmd, cwd=ui_dir)

    except KeyboardInterrupt:
        print("\n⏹️ App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {str(e)}")
        print("Make sure you have all dependencies installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
