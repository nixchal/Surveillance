import shutil
import os
import sys

def copy_dir(src, dst):
    if os.path.exists(dst):
        print(f"Removing existing {dst}")
        shutil.rmtree(dst)
    
    print(f"Copying {src} to {dst}")
    try:
        shutil.copytree(src, dst)
        print(f"Successfully copied to {dst}")
    except Exception as e:
        print(f"Error copying to {dst}: {e}")

base_dir = os.getcwd()
sam3_src = os.path.join(base_dir, "sam3-main", "sam3-main", "sam3")
sam3_dst = os.path.join(base_dir, "sam3")

assets_src = os.path.join(base_dir, "sam3-main", "sam3-main", "assets")
assets_dst = os.path.join(base_dir, "assets")

if not os.path.exists(sam3_src):
    print(f"Source not found: {sam3_src}")
else:
    copy_dir(sam3_src, sam3_dst)

if not os.path.exists(assets_src):
    print(f"Source not found: {assets_src}")
else:
    copy_dir(assets_src, assets_dst)

# Verify
if os.path.exists(sam3_dst) and os.path.exists(assets_dst):
    print("Verification: Directories exist.")
    try:
        import sam3
        print("Verification: sam3 imported successfully.")
    except ImportError as e:
        print(f"Verification: Failed to import sam3: {e}")
else:
    print("Verification: Directories do NOT exist.")
