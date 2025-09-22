"""
Real-time blink detection using the trained machine learning model.
Processes EEG data streams and detects blinks in real-time.
"""

import numpy as np
import pandas as pd
import joblib
import os
from collections import deque
import time

class BlinkDetector:
    def __init__(self, models_dir="models", window_size=15, confidence_threshold=0.5):
        """
        Initialize the blink detector.
        
        Args:
            models_dir: Directory containing trained models
            window_size: Size of the sliding window for feature extraction
            confidence_threshold: Minimum confidence for blink detection
        """
        self.models_dir = models_dir
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # Data buffer for sliding window
        self.data_buffer = deque(maxlen=window_size)
        self.timestamp_buffer = deque(maxlen=window_size)
        
        # Load trained model and scaler
        self.load_model()
        
        # Detection statistics
        self.total_predictions = 0
        self.blink_detections = 0
        self.last_blink_time = 0
        self.min_blink_interval = 0.3  # Minimum time between blinks (seconds)
        
    def load_model(self):
        """Load the trained model and scaler."""
        try:
            # Try to load improved model first, fallback to original
            model_paths = [
                os.path.join(self.models_dir, 'improved_blink_detector.joblib'),
                os.path.join(self.models_dir, 'blink_detector.joblib')
            ]
            
            scaler_paths = [
                os.path.join(self.models_dir, 'improved_scaler.joblib'),
                os.path.join(self.models_dir, 'scaler.joblib')
            ]
            
            metadata_paths = [
                os.path.join(self.models_dir, 'improved_model_metadata.joblib'),
                os.path.join(self.models_dir, 'model_metadata.joblib')
            ]
            
            model_path = model_paths[0] if os.path.exists(model_paths[0]) else model_paths[1]
            scaler_path = scaler_paths[0] if os.path.exists(scaler_paths[0]) else scaler_paths[1]
            metadata_path = metadata_paths[0] if os.path.exists(metadata_paths[0]) else metadata_paths[1]
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.metadata = joblib.load(metadata_path)
            
            print(f"Loaded model: {self.metadata['model_name']}")
            print(f"Model type: {self.metadata['model_type']}")
            print(f"Training score: {self.metadata['best_score']:.4f}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def extract_features(self, values):
        """Extract features from a window of EEG values."""
        if len(values) != self.window_size:
            return None
        
        values = np.array(values)
        
        # Statistical features
        features = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'range': np.max(values) - np.min(values),
            'skewness': pd.Series(values).skew() if len(values) > 1 else 0,
            'kurtosis': pd.Series(values).kurtosis() if len(values) > 1 else 0,
            'rms': np.sqrt(np.mean(values**2)),
            'variance': np.var(values),
        }
        
        # Time-domain features
        features['slope'] = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
        features['energy'] = np.sum(values**2)
        
        # Frequency domain features
        fft_values = np.fft.fft(values)
        features['dominant_freq'] = np.argmax(np.abs(fft_values))
        features['spectral_energy'] = np.sum(np.abs(fft_values)**2)
        
        return features
    
    def add_sample(self, timestamp, value):
        """Add a new EEG sample to the buffer."""
        self.data_buffer.append(value)
        self.timestamp_buffer.append(timestamp)
        
        # Return True if buffer is full and ready for prediction
        return len(self.data_buffer) == self.window_size
    
    def predict_blink(self):
        """Predict if current window contains a blink."""
        if len(self.data_buffer) < self.window_size:
            return False, 0.0
        
        # Extract features
        features = self.extract_features(list(self.data_buffer))
        if features is None:
            return False, 0.0
        
        # Convert to DataFrame and normalize
        feature_df = pd.DataFrame([features])
        feature_normalized = self.scaler.transform(feature_df)
        
        # Make prediction
        prediction = self.model.predict(feature_normalized)[0]
        
        # Get prediction probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(feature_normalized)[0]
            # Find the probability for 'blink' class
            classes = self.model.classes_
            blink_idx = np.where(classes == 'blink')[0]
            if len(blink_idx) > 0:
                confidence = probabilities[blink_idx[0]]
            else:
                confidence = 0.0
        else:
            confidence = 1.0 if prediction == 'blink' else 0.0
        
        self.total_predictions += 1
        
        # Check if it's a blink with sufficient confidence
        is_blink = (prediction == 'blink' and confidence >= self.confidence_threshold)
        
        # Apply minimum interval filter
        current_time = time.time()
        if is_blink and (current_time - self.last_blink_time) < self.min_blink_interval:
            is_blink = False
        
        if is_blink:
            self.blink_detections += 1
            self.last_blink_time = current_time
        
        return is_blink, confidence
    
    def process_sample(self, timestamp, value):
        """Process a single EEG sample and return blink detection result."""
        buffer_ready = self.add_sample(timestamp, value)
        
        if buffer_ready:
            return self.predict_blink()
        else:
            return False, 0.0
    
    def reset_buffer(self):
        """Reset the data buffer."""
        self.data_buffer.clear()
        self.timestamp_buffer.clear()
    
    def get_statistics(self):
        """Get detection statistics."""
        if self.total_predictions > 0:
            detection_rate = self.blink_detections / self.total_predictions
        else:
            detection_rate = 0.0
        
        return {
            'total_predictions': self.total_predictions,
            'blink_detections': self.blink_detections,
            'detection_rate': detection_rate,
            'last_blink_time': self.last_blink_time
        }
    
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for blink detection."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Confidence threshold set to {self.confidence_threshold:.3f}")
    
    def set_min_blink_interval(self, interval):
        """Set the minimum interval between blinks."""
        self.min_blink_interval = max(0.1, interval)
        print(f"Minimum blink interval set to {self.min_blink_interval:.3f} seconds")

class RealTimeBlinkDetector:
    def __init__(self, eeg_simulator, models_dir="models"):
        """
        Real-time blink detector that works with EEG simulator.
        
        Args:
            eeg_simulator: EEGSimulator instance
            models_dir: Directory containing trained models
        """
        self.eeg_simulator = eeg_simulator
        self.detector = BlinkDetector(models_dir=models_dir, window_size=15)
        self.is_running = False
        
        # Callbacks for blink events
        self.blink_callbacks = []
        
    def add_blink_callback(self, callback):
        """Add a callback function to be called when a blink is detected."""
        self.blink_callbacks.append(callback)
    
    def remove_blink_callback(self, callback):
        """Remove a blink callback."""
        if callback in self.blink_callbacks:
            self.blink_callbacks.remove(callback)
    
    def on_blink_detected(self, confidence):
        """Called when a blink is detected."""
        for callback in self.blink_callbacks:
            try:
                callback(confidence)
            except Exception as e:
                print(f"Error in blink callback: {e}")
    
    def start(self):
        """Start real-time blink detection."""
        if not self.eeg_simulator.is_running:
            self.eeg_simulator.start()
        
        self.is_running = True
        print("Real-time blink detection started")
    
    def stop(self):
        """Stop real-time blink detection."""
        self.is_running = False
        print("Real-time blink detection stopped")
    
    def update(self):
        """Update the detector with latest EEG data. Call this regularly."""
        if not self.is_running:
            return False, 0.0
        
        # Get latest sample from simulator
        sample = self.eeg_simulator.get_latest_sample()
        if sample is None:
            return False, 0.0
        
        # Process the sample
        is_blink, confidence = self.detector.process_sample(
            sample['timestamp'], 
            sample['value']
        )
        
        # Trigger callbacks if blink detected
        if is_blink:
            self.on_blink_detected(confidence)
        
        return is_blink, confidence
    
    def get_statistics(self):
        """Get detection statistics."""
        return self.detector.get_statistics()

if __name__ == "__main__":
    # Test the blink detector
    from eeg_simulator import EEGSimulator
    
    # Create simulator and detector
    simulator = EEGSimulator(sampling_rate=100)
    detector = RealTimeBlinkDetector(simulator)
    
    # Add a callback to print when blinks are detected
    def on_blink(confidence):
        print(f"BLINK DETECTED! Confidence: {confidence:.3f}")
    
    detector.add_blink_callback(on_blink)
    
    try:
        # Start detection
        detector.start()
        
        print("Real-time blink detection running...")
        print("Press Ctrl+C to stop")
        
        # Manually trigger some blinks for testing
        import threading
        def trigger_blinks():
            time.sleep(2)
            simulator.trigger_blink()
            time.sleep(3)
            simulator.trigger_blink()
            time.sleep(2)
            simulator.trigger_blink()
        
        blink_thread = threading.Thread(target=trigger_blinks)
        blink_thread.daemon = True
        blink_thread.start()
        
        # Main detection loop
        for i in range(1000):  # Run for 10 seconds
            is_blink, confidence = detector.update()
            time.sleep(0.01)  # 10ms update rate
        
        # Print statistics
        stats = detector.get_statistics()
        print(f"\nDetection Statistics:")
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Blinks detected: {stats['blink_detections']}")
        print(f"Detection rate: {stats['detection_rate']:.4f}")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()
        simulator.stop()