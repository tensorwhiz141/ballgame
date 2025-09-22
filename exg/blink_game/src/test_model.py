"""
Test script for the improved blink detection model.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eeg_simulator import EEGSimulator
from src.blink_detector import RealTimeBlinkDetector
import time

def test_model():
    """Test the blink detection model."""
    print("Testing improved blink detection model...")
    
    # Check if model exists
    model_paths = [
        "models/improved_blink_detector.joblib",
        "models/blink_detector.joblib"
    ]
    
    model_exists = any(os.path.exists(path) for path in model_paths)
    
    if not model_exists:
        print("Error: Trained model not found.")
        return
    
    # Create simulator and detector
    simulator = EEGSimulator(sampling_rate=100)
    detector = RealTimeBlinkDetector(simulator)
    
    # Add callback for detected blinks
    def on_blink(confidence):
        print(f"BLINK DETECTED! Confidence: {confidence:.3f}")
    
    detector.add_blink_callback(on_blink)
    detector.start()
    
    print("Blink detector running for 10 seconds...")
    print("Triggering test blinks...")
    
    # Trigger test blinks
    time.sleep(2)
    simulator.trigger_blink()
    print("Triggered first blink")
    
    time.sleep(3)
    simulator.trigger_blink()
    print("Triggered second blink")
    
    time.sleep(3)
    simulator.trigger_blink()
    print("Triggered third blink")
    
    time.sleep(2)
    
    # Update detector
    for i in range(1000):
        detector.update()
        time.sleep(0.01)
    
    # Print statistics
    stats = detector.get_statistics()
    print(f"\nDetection Statistics:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Blinks detected: {stats['blink_detections']}")
    if stats['total_predictions'] > 0:
        detection_rate = stats['blink_detections'] / stats['total_predictions']
        print(f"Detection rate: {detection_rate:.4f}")
    
    detector.stop()
    simulator.stop()
    print("Test completed!")

if __name__ == "__main__":
    test_model()