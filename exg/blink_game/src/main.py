"""
Main entry point for the EEG Blink-Controlled Game.
Provides options to train models, test components, or run the game.
"""

import sys
import os
import argparse

# Change to the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(script_dir))

def train_model():
    """Train the blink detection model."""
    print("Training blink detection model...")
    from model_training import BlinkDetectionTrainer
    
    trainer = BlinkDetectionTrainer()
    model, scaler, accuracy, f1 = trainer.train_complete_pipeline()
    
    print(f"\nModel training completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Final F1-score: {f1:.4f}")
    print("Model saved in 'models/' directory")

def train_improved_model():
    """Train the improved blink detection model."""
    print("Training improved blink detection model...")
    from src.improved_model_training import ImprovedBlinkDetectionTrainer
    
    trainer = ImprovedBlinkDetectionTrainer()
    model, scaler, accuracy, f1 = trainer.train_complete_pipeline()
    
    print(f"\nImproved model training completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Final F1-score: {f1:.4f}")
    print("Model saved in 'models/' directory")

def test_eeg_simulator():
    """Test the EEG simulator."""
    print("Testing EEG simulator...")
    from src.eeg_simulator import EEGSimulator
    import time
    
    simulator = EEGSimulator(sampling_rate=100)
    simulator.start()
    
    print("EEG simulator running for 5 seconds...")
    print("Triggering test blinks...")
    
    # Trigger some test blinks
    time.sleep(1)
    simulator.trigger_blink()
    time.sleep(2)
    simulator.trigger_blink()
    time.sleep(2)
    
    simulator.stop()
    print("EEG simulator test completed!")

def test_blink_detector():
    """Test the blink detector."""
    print("Testing blink detector...")
    from src.eeg_simulator import EEGSimulator
    from src.blink_detector import RealTimeBlinkDetector
    import time
    
    # Check if model exists
    model_paths = [
        "models/improved_blink_detector.joblib",
        "models/blink_detector.joblib"
    ]
    
    model_exists = any(os.path.exists(path) for path in model_paths)
    
    if not model_exists:
        print("Error: Trained model not found. Please run 'python main.py --train' first.")
        return
    
    simulator = EEGSimulator(sampling_rate=100)
    detector = RealTimeBlinkDetector(simulator)
    
    def on_blink(confidence):
        print(f"BLINK DETECTED! Confidence: {confidence:.3f}")
    
    detector.add_blink_callback(on_blink)
    detector.start()
    
    print("Blink detector running for 10 seconds...")
    print("Triggering test blinks...")
    
    # Trigger test blinks
    time.sleep(2)
    simulator.trigger_blink()
    time.sleep(3)
    simulator.trigger_blink()
    time.sleep(3)
    simulator.trigger_blink()
    time.sleep(2)
    
    # Update detector
    for _ in range(1000):
        detector.update()
        time.sleep(0.01)
    
    stats = detector.get_statistics()
    print(f"\nDetection Statistics:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Blinks detected: {stats['blink_detections']}")
    
    detector.stop()
    simulator.stop()
    print("Blink detector test completed!")

def run_game():
    """Run the main game."""
    print("Starting EEG Blink-Controlled Game...")
    
    # Check if model exists
    model_paths = [
        "models/improved_blink_detector.joblib",
        "models/blink_detector.joblib"
    ]
    
    model_exists = any(os.path.exists(path) for path in model_paths)
    
    if not model_exists:
        print("Error: Trained model not found. Please run 'python main.py --train' first.")
        return
    
    from src.game import Game
    
    try:
        game = Game()
        game.run()
    except Exception as e:
        print(f"Error running game: {e}")
        print("Make sure pygame is properly installed and display is available.")

def process_data():
    """Process the EEG datasets."""
    print("Processing EEG datasets...")
    from src.data_processing import EEGDataProcessor
    
    processor = EEGDataProcessor(data_dir="data")
    X_train, X_test, y_train, y_test, scaler = processor.process_all()
    
    print("Data processing completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="EEG Blink-Controlled Game")
    parser.add_argument("--train", action="store_true", help="Train the blink detection model")
    parser.add_argument("--train-improved", action="store_true", help="Train the improved blink detection model")
    parser.add_argument("--test-eeg", action="store_true", help="Test EEG simulator")
    parser.add_argument("--test-detector", action="store_true", help="Test blink detector")
    parser.add_argument("--process-data", action="store_true", help="Process EEG datasets")
    parser.add_argument("--game", action="store_true", help="Run the game")
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.train_improved:
        train_improved_model()
    elif args.test_eeg:
        test_eeg_simulator()
    elif args.test_detector:
        test_blink_detector()
    elif args.process_data:
        process_data()
    elif args.game:
        run_game()
    else:
        # Interactive menu
        print("\n" + "="*50)
        print("EEG BLINK-CONTROLLED GAME")
        print("="*50)
        print("\nWelcome! This game uses EEG signals to detect blinks")
        print("and control a jumping ball to avoid obstacles.")
        print("\nWhat would you like to do?")
        print("1. Train the blink detection model")
        print("2. Train the improved blink detection model")
        print("3. Test EEG simulator")
        print("4. Test blink detector")
        print("5. Process EEG data")
        print("6. Run the game")
        print("7. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-7): ").strip()
                
                if choice == "1":
                    train_model()
                    break
                elif choice == "2":
                    train_improved_model()
                    break
                elif choice == "3":
                    test_eeg_simulator()
                    break
                elif choice == "4":
                    test_blink_detector()
                    break
                elif choice == "5":
                    process_data()
                    break
                elif choice == "6":
                    run_game()
                    break
                elif choice == "7":
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1-7.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()