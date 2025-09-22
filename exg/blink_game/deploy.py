"""
Deployment script to create a portable package of the EEG Blink Game.
This creates a zip file that can be easily transferred to another computer.
"""

import os
import shutil
import zipfile
import sys
from datetime import datetime

def create_deployment_package():
    """Create a portable deployment package."""
    
    print("Creating EEG Blink Game deployment package...")
    
    # Create deployment directory
    deploy_dir = "blink_game_portable"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Files and directories to include
    items_to_copy = [
        "src/",
        "data/",
        "models/",
        "requirements.txt",
        "README.md",
        "PICO_SETUP.md"
    ]
    
    # Copy files
    for item in items_to_copy:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.copytree(item, os.path.join(deploy_dir, item))
                print(f"Copied directory: {item}")
            else:
                shutil.copy2(item, deploy_dir)
                print(f"Copied file: {item}")
        else:
            print(f"Warning: {item} not found, skipping...")
    
    # Create setup script for the target computer
    setup_script = """@echo off
echo Setting up EEG Blink Game...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found, creating virtual environment...
python -m venv blink_game_env

echo Activating virtual environment...
call blink_game_env\\Scripts\\activate

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete! 
echo.
echo To run the game:
echo 1. Open command prompt in this folder
echo 2. Run: blink_game_env\\Scripts\\activate
echo 3. Run: python src/main.py
echo.
echo Or simply double-click run_game.bat
echo.
pause
"""
    
    with open(os.path.join(deploy_dir, "setup.bat"), "w") as f:
        f.write(setup_script)
    
    # Create run script
    run_script = """@echo off
echo Starting EEG Blink Game...
call blink_game_env\\Scripts\\activate
python src/main.py
pause
"""
    
    with open(os.path.join(deploy_dir, "run_game.bat"), "w") as f:
        f.write(run_script)
    
    # Create Linux/Mac setup script
    setup_script_unix = """#!/bin/bash
echo "Setting up EEG Blink Game..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "Python found, creating virtual environment..."
python3 -m venv blink_game_env

echo "Activating virtual environment..."
source blink_game_env/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "Setup complete!"
echo
echo "To run the game:"
echo "1. Open terminal in this folder"
echo "2. Run: source blink_game_env/bin/activate"
echo "3. Run: python src/main.py"
echo
echo "Or run: ./run_game.sh"
echo
"""
    
    with open(os.path.join(deploy_dir, "setup.sh"), "w") as f:
        f.write(setup_script_unix)
    
    # Make it executable
    os.chmod(os.path.join(deploy_dir, "setup.sh"), 0o755)
    
    # Create run script for Unix
    run_script_unix = """#!/bin/bash
echo "Starting EEG Blink Game..."
source blink_game_env/bin/activate
python src/main.py
"""
    
    with open(os.path.join(deploy_dir, "run_game.sh"), "w") as f:
        f.write(run_script_unix)
    
    os.chmod(os.path.join(deploy_dir, "run_game.sh"), 0o755)
    
    # Create deployment README
    deploy_readme = """# EEG Blink-Controlled Game - Portable Version

This is a portable version of the EEG Blink-Controlled Game that can be run on any computer.

## Quick Start

### Windows:
1. Double-click `setup.bat` to install dependencies
2. Double-click `run_game.bat` to start the game

### Linux/Mac:
1. Run `./setup.sh` to install dependencies
2. Run `./run_game.sh` to start the game

## Manual Setup

If the automatic setup doesn't work:

1. Install Python 3.8+ from https://python.org
2. Open terminal/command prompt in this folder
3. Create virtual environment: `python -m venv blink_game_env`
4. Activate it:
   - Windows: `blink_game_env\\Scripts\\activate`
   - Linux/Mac: `source blink_game_env/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Run the game: `python src/main.py`

## Game Features

- **Simulated Mode**: Uses realistic EEG data simulation for testing
- **Real Hardware Mode**: Can connect to Raspberry Pi Pico for real EEG input
- **Machine Learning**: Trained model detects blinks in real-time
- **Interactive Game**: Jump over obstacles by blinking!

## How to Play

1. Start the game using one of the methods above
2. Choose option 5 "Run the game" from the menu
3. The ball moves automatically from left to right
4. Blink (or press SPACE) to make the ball jump over red obstacles
5. Avoid collisions to keep playing and increase your score!

## Controls

- **Blink**: Make the ball jump (main control)
- **SPACE**: Manual jump (backup control)
- **B**: Trigger test blink (simulation mode only)
- **P**: Pause/unpause game
- **R**: Restart game (when game over)
- **Q**: Quit game

## Connecting Real Hardware

See `PICO_SETUP.md` for detailed instructions on connecting a Raspberry Pi Pico with EEG sensors.

To use real hardware:
```
python src/game_with_pico.py --pico --port COM3
```

## Troubleshooting

1. **Python not found**: Install Python 3.8+ from python.org
2. **Permission errors**: Run as administrator (Windows) or use sudo (Linux/Mac)
3. **Game won't start**: Check that all dependencies installed correctly
4. **No display**: Make sure you have a graphical environment (not SSH)

## System Requirements

- Python 3.8 or higher
- Windows 10+, macOS 10.14+, or Linux
- 4GB RAM minimum
- Graphics support for pygame
- USB port (if using real Pico hardware)

## Files Included

- `src/`: Source code files
- `data/`: EEG training datasets
- `models/`: Trained machine learning models
- `requirements.txt`: Python dependencies
- `README.md`: Main documentation
- `PICO_SETUP.md`: Hardware setup guide

Enjoy playing the EEG Blink-Controlled Game!
"""
    
    with open(os.path.join(deploy_dir, "DEPLOYMENT_README.md"), "w") as f:
        f.write(deploy_readme)
    
    # Create zip file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"blink_game_portable_{timestamp}.zip"
    
    print(f"Creating zip file: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, deploy_dir)
                zipf.write(file_path, arc_path)
                
    print(f"Deployment package created: {zip_filename}")
    print(f"Package size: {os.path.getsize(zip_filename) / (1024*1024):.1f} MB")
    
    # Clean up temporary directory
    shutil.rmtree(deploy_dir)
    
    print("\nDeployment complete!")
    print(f"Send '{zip_filename}' to your friend's laptop.")
    print("They should extract it and run setup.bat (Windows) or setup.sh (Linux/Mac)")
    
    return zip_filename

if __name__ == "__main__":
    try:
        zip_file = create_deployment_package()
        print(f"\nSuccess! Created: {zip_file}")
    except Exception as e:
        print(f"Error creating deployment package: {e}")
        sys.exit(1)
