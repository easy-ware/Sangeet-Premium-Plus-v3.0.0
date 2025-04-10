
import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path


def check_venv(venv_path):
    """
    Check if a virtual environment at the given path has all crucial components.
    If something important is missing, delete the venv but still return True.
    Return True always to indicate check was completed, whether venv was valid or deleted.
    
    Args:
        venv_path (str): Path to the virtual environment directory
    
    Returns:
        bool: Always returns True after checking and taking appropriate action
    """
    venv_path = Path(venv_path).resolve()
    
    # Check if directory exists
    if not venv_path.exists() or not venv_path.is_dir():
        print(f"Error: {venv_path} does not exist or is not a directory")
        return True  # Return True even though check failed
    
    # Determine platform-specific components
    is_windows = platform.system() == "Windows"
    is_posix = os.name == "posix"  # Linux, macOS, etc.
    
    # Define critical components based on platform
    critical_components = []
    
    if is_windows:
        critical_components = [
            venv_path / "Scripts",
            venv_path / "Scripts" / "python.exe",
            venv_path / "Scripts" / "pip.exe",
            venv_path / "Lib",
            venv_path / "Lib" / "site-packages",
            venv_path / "pyvenv.cfg",
        ]
    elif is_posix:
        critical_components = [
            venv_path / "bin",
            venv_path / "bin" / "python",
            venv_path / "bin" / "pip",
            venv_path / "lib",
            venv_path / "pyvenv.cfg",
        ]
        
        # Find the Python version directory (like python3.9, python3.10, etc.)
        lib_path = venv_path / "lib"
        if lib_path.exists():
            python_dirs = list(lib_path.glob("python*"))
            if python_dirs:
                critical_components.append(python_dirs[0])
                critical_components.append(python_dirs[0] / "site-packages")
    
    # Check for missing components
    missing_components = [comp for comp in critical_components if not comp.exists()]
    
    if missing_components:
        print(f"Virtual environment at {venv_path} is missing critical components:")
        for component in missing_components:
            print(f"  - {component}")
        
        # Check if the venv is actually a venv by looking for pyvenv.cfg
        if not (venv_path / "pyvenv.cfg").exists():
            print(f"Error: {venv_path} does not appear to be a valid virtual environment")
            
        # Delete the virtual environment
        print(f"Deleting invalid virtual environment at {venv_path}")
        try:
            shutil.rmtree(venv_path)
            print(f"Successfully deleted {venv_path}")
        except Exception as e:
            print(f"Error deleting virtual environment: {e}")
        
        return True  # Return True even though venv was invalid and deleted
    
    # Check if Python and pip are functional in the venv
    try:
        # Construct the path to the Python executable
        if is_windows:
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        # Test Python execution
        result = subprocess.run(
            [str(python_exe), "-c", "import sys; print('Python is working')"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Error: Python in the virtual environment is not functioning properly")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            print(f"Deleting invalid virtual environment at {venv_path}")
            shutil.rmtree(venv_path)
            return True  # Return True even though Python test failed and venv was deleted
        
    except Exception as e:
        print(f"Error testing Python in the virtual environment: {e}")
        print(f"Deleting invalid virtual environment at {venv_path}")
        shutil.rmtree(venv_path)
        return True  # Return True even though there was an exception and venv was deleted
    
    print(f"Virtual environment at {venv_path} is valid and contains all necessary components")
    return True  # Return True because venv is valid


def main():
    if len(sys.argv) < 2:
        print("Usage: python venv_checker.py <path_to_venv>")
        sys.exit(0)  # Always exit with success code
    
    venv_path = sys.argv[1]
    check_venv(venv_path)  # Result not used as we always exit with success
    
    sys.exit(0)  # Always exit with success code


if __name__ == "__main__":
    main()
