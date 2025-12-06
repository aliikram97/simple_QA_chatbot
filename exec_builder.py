# build.py - Cross-platform build script
import os
import sys
import subprocess
import shutil


def clean_build():
    """Remove previous build artifacts"""
    dirs_to_remove = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Cleaned {dir_name}/")

    # Remove spec files
    for file in os.listdir('.'):
        if file.endswith('.spec'):
            os.remove(file)
            print(f"Removed {file}")


def build_executable():
    """Build executable using PyInstaller"""

    # Your main script name
    script_name = "chatbot.py"  # CHANGE THIS to your script name
    app_name = "MyPipeline"  # CHANGE THIS to your app name

    # Detect platform
    is_windows = sys.platform == "win32"
    is_mac = sys.platform == "darwin"

    print(f"Building for {sys.platform}...")

    # Base PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--clean",
        f"--name={app_name}",
    ]

    # Add icon if available
    if is_windows and os.path.exists("icon.ico"):
        cmd.append("--icon=icon.ico")
    elif is_mac and os.path.exists("icon.icns"):
        cmd.append("--icon=icon.icns")

    # For GUI apps (no console window)
    # Uncomment the next line if you have a GUI app
    # cmd.append("--windowed")

    # Add hidden imports if needed (common dependencies)
    # cmd.extend(["--hidden-import=numpy", "--hidden-import=pandas"])

    # Add data files if needed
    # cmd.append("--add-data=config.json:.")

    cmd.append(script_name)

    # Run PyInstaller
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Build successful!")
        print(f"Executable location: dist/{app_name}")

        if is_mac:
            print("\nTo create a .app bundle, run:")
            print(f"pyinstaller --onefile --windowed --name={app_name} {script_name}")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed: {e}")
        sys.exit(1)


def main():
    print("=== Cross-Platform Python Build Script ===\n")

    # Install PyInstaller if not present
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    # Clean and build
    clean_build()
    build_executable()


if __name__ == "__main__":
    main()