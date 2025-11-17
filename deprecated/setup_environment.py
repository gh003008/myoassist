"""
Setup MyoAssist Environment
============================

Installs all required dependencies for MyoAssist Imitation Learning

Usage:
    python setup_environment.py
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code: {e.returncode}")
        return False


def main():
    print("\n" + "="*80)
    print("🚀 MyoAssist Environment Setup")
    print("="*80)
    
    print("\nThis script will install all required dependencies.")
    print("Make sure you're in the correct conda/virtual environment!")
    
    # Check Python version
    print(f"\n📍 Python version: {sys.version}")
    print(f"📍 Python executable: {sys.executable}")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Step 1: Install core dependencies
    success = run_command(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        "Installing MyoAssist package"
    )
    
    if not success:
        print("\n⚠️  Warning: Package installation had issues, but continuing...")
    
    # Step 2: Install additional dependencies
    additional_packages = [
        "gymnasium",
        "stable-baselines3",
        "torch",
        "numpy",
        "matplotlib",
        "mediapy",
    ]
    
    success = run_command(
        [sys.executable, "-m", "pip", "install"] + additional_packages,
        "Installing additional packages"
    )
    
    if not success:
        print("\n❌ Failed to install additional packages")
        sys.exit(1)
    
    # Step 3: Verify installation
    print("\n" + "="*80)
    print("🔍 Verifying Installation")
    print("="*80)
    
    try:
        import gymnasium
        import stable_baselines3
        import torch
        import numpy as np
        import matplotlib
        
        print("\n✅ All core packages imported successfully!")
        print(f"   - gymnasium: {gymnasium.__version__}")
        print(f"   - stable-baselines3: {stable_baselines3.__version__}")
        print(f"   - torch: {torch.__version__}")
        print(f"   - numpy: {np.__version__}")
        print(f"   - matplotlib: {matplotlib.__version__}")
        
    except ImportError as e:
        print(f"\n❌ Import verification failed: {e}")
        sys.exit(1)
    
    # Step 4: Run verification script
    print("\n" + "="*80)
    print("🧪 Running Environment Verification")
    print("="*80)
    
    success = run_command(
        [sys.executable, "verify_S004_setup.py"],
        "Verifying S004 setup"
    )
    
    if success:
        print("\n" + "="*80)
        print("✅ Environment setup completed successfully!")
        print("="*80)
        print("\n🎉 You're ready to start training!")
        print("\nQuick start:")
        print("  python train_S004_motion.py --quick_test")
        print("\nFull training:")
        print("  python train_S004_motion.py")
        print("="*80 + "\n")
    else:
        print("\n⚠️  Setup completed but verification had issues.")
        print("Please check the error messages above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(0)
