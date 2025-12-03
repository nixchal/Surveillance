import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, login

def download_sam3():
    # Define paths
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models" / "sam3"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "sam3.pt"
    config_path = models_dir / "config.json"

    print(f"Target directory: {models_dir}")

    if checkpoint_path.exists():
        print(f"✅ SAM 3.0 checkpoint already exists at: {checkpoint_path}")
        return

    print("❌ SAM 3.0 checkpoint not found locally.")
    print("This model is GATED, meaning you need to accept the license on Hugging Face and have a valid token.")
    
    token = input("Enter your Hugging Face Access Token (hidden): ").strip()
    if not token:
        print("No token provided. Please run 'huggingface-cli login' or provide a token.")
        return

    try:
        print("Logging in...")
        login(token=token)
        
        print("Downloading config.json...")
        hf_hub_download(repo_id="facebook/sam3", filename="config.json", local_dir=models_dir, local_dir_use_symlinks=False)
        
        print("Downloading sam3.pt (this may take a while)...")
        hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_dir=models_dir, local_dir_use_symlinks=False)
        
        print("✅ Download complete!")
        print(f"Files saved to: {models_dir}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Please ensure you have accepted the license at https://huggingface.co/facebook/sam3")

if __name__ == "__main__":
    download_sam3()
