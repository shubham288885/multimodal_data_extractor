import os

def create_init_files(root_dir):
    """
    Creates __init__.py files in all directories in the project
    to make Python properly recognize the directory structure as packages.
    
    Args:
        root_dir: The root directory to start from
    """
    print(f"Creating __init__.py files in {root_dir} and subdirectories...")
    
    # Skip these directories
    skip_dirs = [".git", "__pycache__", "venv", "env", ".venv", "fresh_env", "static"]
    
    # Count of files created
    count = 0
    
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip directories we want to exclude
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        
        # Check if __init__.py already exists
        init_file = os.path.join(dirpath, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Auto-generated __init__.py file\n")
            print(f"Created: {init_file}")
            count += 1
    
    print(f"Done! Created {count} __init__.py files.")

if __name__ == "__main__":
    # Start from current directory
    create_init_files(os.path.dirname(os.path.abspath(__file__))) 