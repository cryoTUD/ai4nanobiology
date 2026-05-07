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


def check_files_are_readable():
    GNN_DATA_FOLDER = "/projects/nb4170/gnn_data"

    # Check the root folder first
    if not check_path(GNN_DATA_FOLDER):
        print("  Root data folder missing. Skipping further checks.")
        return False

    files_to_check = [
        "1L2Y.pdb", "1L2Y.cif", "4AKE.pdb", "2ECK.pdb", "df_pscdb.csv",
    ]
    subsubfolders = ["train", "val", "test"]

    failures = 0
    for file in files_to_check:
        if not check_path(os.path.join(GNN_DATA_FOLDER, file)):
            failures += 1

    for sub in subsubfolders:
        if not check_path(os.path.join(GNN_DATA_FOLDER, "all_datasets", sub)):
            failures += 1

    total = len(files_to_check) + len(subsubfolders)
    print(f"\n  {total - failures} of {total} paths accessible.")
    return failures == 0


def check_everything():
    print("Checking imports...")
    imports_ok = check_imports()

    print("\nChecking file readability...")
    files_ok = check_files_are_readable()

    print("\n" + "="*40)
    if imports_ok and files_ok:
        print("All checks passed. You're ready to go.")
    else:
        print("Some checks failed. See messages above.")
    