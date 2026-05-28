import os

def try_to_import(module_name):
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_imports():
    modules_to_check = [
        "numpy", "pandas", "torch", "torch_geometric",
        "gemmi", "plotly", "graphein", "matplotlib",
    ]
    failures = []
    for module in modules_to_check:
        ok, err = try_to_import(module)
        if ok:
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}: {err}")
            failures.append(module)

    print(f"\n  {len(modules_to_check) - len(failures)} of {len(modules_to_check)} imports succeeded.")
    return len(failures) == 0


def check_path(path, label="path"):
    """Check whether a path exists and is readable. Returns True if ok."""
    if not os.path.exists(path):
        print(f"  ✗ Missing: {path}")
        return False
    if not os.access(path, os.R_OK):
        print(f"  ✗ Not readable (permissions issue): {path}")
        return False
    print(f"  ✓ {path}")
    return True

def check_gpu_available():
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("  ✗ No GPU available. Please check your environment.")
        return False
    
def check_files_are_readable():
    GNN_DATA_FOLDER = "/projects/nb4170/week_5/saved_models/"

    # Check the root folder first
    if not check_path(GNN_DATA_FOLDER):
        print("  Root data folder missing. Skipping further checks.")
        return False

def check_folder_is_writable(folder_path):
    from datetime import datetime
    if not os.path.exists(folder_path):
        print(f"  ✗ Folder does not exist: {folder_path}")
        return False
    if not os.access(folder_path, os.W_OK):
        print(f"  ✗ Not writable (permissions issue): {folder_path}")
        return False
    # Create a temporary file to test writability
    username = os.getenv("USER", "unknown_user")
    temp_file_name = f"temp_write_test_{username}_{datetime.now().timestamp()}.tmp"
    temp_file_path = os.path.join(folder_path, temp_file_name)
    try:
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write("This is a test.")
        os.remove(temp_file_path)  # Clean up
    except Exception as e:
        print(f"  ✗ Failed to write to folder: {folder_path}. Error: {e}")
        return False
    print(f"  ✓ Writable: {folder_path}")
    return True

def check_everything():
    print("Checking imports...")
    imports_ok = check_imports()

    print("\nChecking GPU availability...")
    gpu_ok = check_gpu_available()

    print("\nChecking file readability...")
    files_ok = check_files_are_readable()

    print("\nChecking output folder writability...")
    output_folder = "/projects/nb4170/week_5/tinyLMs/"
    output_ok = check_folder_is_writable(output_folder)

    print("\n" + "="*40)
    if imports_ok and files_ok and output_ok and gpu_ok:
        print("All checks passed. You're ready to go.")
    else:
        print("Some checks failed. See messages above.")
    