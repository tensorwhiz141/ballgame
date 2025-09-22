# EEG Blink-Controlled Game

A real-time game where a ball jumps over obstacles when the player blinks, using EEG signal processing and machine learning.

## Project Structure

```
blink_game/
├── data/                    # Dataset files
├── models/                  # Trained ML models
├── src/                     # Source code
│   ├── data_processing.py   # Data preprocessing and feature extraction
│   ├── model_training.py    # ML model training
│   ├── eeg_simulator.py     # Real-time EEG data simulation
│   ├── blink_detector.py    # Real-time blink detection
│   ├── game.py             # Main game implementation
│   └── main.py             # Main application entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

1. Activate the virtual environment:
   ```
   blink_game_env\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the game:
   ```
   python src/main.py
   ```

## How it Works

1. **Data Processing**: Combines and preprocesses EEG datasets
2. **Model Training**: Trains a machine learning model to detect blinks
3. **Real-time Simulation**: Simulates EEG data stream
4. **Blink Detection**: Uses trained model to detect blinks in real-time
5. **Game Control**: Detected blinks trigger ball jumps in the game

## Features

- Real-time EEG signal processing
- Machine learning-based blink detection
- Interactive pygame-based game
- Portable deployment package
