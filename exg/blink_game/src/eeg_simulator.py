"""
Real-time EEG data simulation for testing the blink detection system.
Simulates EEG signals with realistic patterns and occasional blinks.
"""

import numpy as np
import pandas as pd
import time
import threading
import queue
from collections import deque
import random

class EEGSimulator:
    def __init__(self, sampling_rate=250, buffer_size=1000):
        """
        Initialize EEG simulator.
        
        Args:
            sampling_rate: Samples per second (Hz)
            buffer_size: Maximum number of samples to keep in buffer
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.is_running = False
        
        # Data buffer for real-time streaming
        self.data_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Queue for communication between threads
        self.data_queue = queue.Queue()
        
        # Simulation parameters
        self.base_frequency = 10  # Alpha wave frequency (Hz)
        self.noise_amplitude = 5000
        self.signal_amplitude = 40000
        self.blink_amplitude = 25000  # Amplitude during blinks
        self.blink_duration = 0.2  # Duration of a blink in seconds
        
        # Blink simulation
        self.blink_probability = 0.02  # Probability of blink per second
        self.current_blink_samples = 0
        self.blink_samples_total = int(self.blink_duration * self.sampling_rate)
        
        # Load real data patterns for more realistic simulation
        self.load_real_patterns()
        
    def load_real_patterns(self):
        """Load real EEG patterns from the datasets for realistic simulation."""
        try:
            # Load the combined dataset
            df1 = pd.read_csv("data/exg_blink_dataset1 (1).csv")
            df2 = pd.read_csv("data/exg_blink_dataset2000.csv")
            combined_data = pd.concat([df1, df2], ignore_index=True)
            
            # Separate normal and blink patterns
            self.normal_patterns = combined_data[combined_data['label'] == 'normal']['value'].values
            self.blink_patterns = combined_data[combined_data['label'] == 'blink']['value'].values
            
            print(f"Loaded {len(self.normal_patterns)} normal patterns and {len(self.blink_patterns)} blink patterns")
            
        except Exception as e:
            print(f"Could not load real patterns: {e}")
            print("Using synthetic patterns instead")
            self.normal_patterns = None
            self.blink_patterns = None
    
    def generate_normal_sample(self, timestamp):
        """Generate a normal EEG sample."""
        if self.normal_patterns is not None and len(self.normal_patterns) > 0:
            # Use real normal pattern with some variation
            base_value = random.choice(self.normal_patterns)
            noise = np.random.normal(0, self.noise_amplitude * 0.1)
            return base_value + noise
        else:
            # Generate synthetic normal EEG signal
            # Alpha wave component
            alpha_wave = self.signal_amplitude * np.sin(2 * np.pi * self.base_frequency * timestamp / 1000)
            
            # Beta wave component (higher frequency, lower amplitude)
            beta_wave = self.signal_amplitude * 0.3 * np.sin(2 * np.pi * 25 * timestamp / 1000)
            
            # Random noise
            noise = np.random.normal(0, self.noise_amplitude)
            
            return alpha_wave + beta_wave + noise + self.signal_amplitude
    
    def generate_blink_sample(self, timestamp, blink_progress):
        """Generate a blink EEG sample."""
        if self.blink_patterns is not None and len(self.blink_patterns) > 0:
            # Use real blink pattern with some variation
            base_value = random.choice(self.blink_patterns)
            noise = np.random.normal(0, self.noise_amplitude * 0.1)
            return base_value + noise
        else:
            # Generate synthetic blink signal (lower amplitude)
            # Blink creates a characteristic dip in the signal
            blink_factor = np.sin(np.pi * blink_progress)  # Bell curve for blink
            normal_signal = self.generate_normal_sample(timestamp)
            blink_component = -self.blink_amplitude * blink_factor
            
            return normal_signal + blink_component
    
    def should_blink(self):
        """Determine if a blink should occur."""
        # Random blink based on probability
        if self.current_blink_samples == 0:
            return random.random() < (self.blink_probability / self.sampling_rate)
        return False
    
    def generate_sample(self, timestamp):
        """Generate a single EEG sample."""
        # Check if we should start a new blink
        if self.should_blink():
            self.current_blink_samples = self.blink_samples_total
        
        # Generate sample based on current state
        if self.current_blink_samples > 0:
            # We're in a blink
            blink_progress = (self.blink_samples_total - self.current_blink_samples) / self.blink_samples_total
            sample = self.generate_blink_sample(timestamp, blink_progress)
            self.current_blink_samples -= 1
            label = 'blink'
        else:
            # Normal state
            sample = self.generate_normal_sample(timestamp)
            label = 'normal'
        
        return sample, label
    
    def simulation_thread(self):
        """Main simulation thread that generates data continuously."""
        start_time = time.time() * 1000  # Convert to milliseconds
        sample_interval = 1000 / self.sampling_rate  # Interval between samples in ms
        
        sample_count = 0
        
        while self.is_running:
            current_time = start_time + (sample_count * sample_interval)
            
            # Generate sample
            value, label = self.generate_sample(current_time)
            
            # Add to buffer
            self.data_buffer.append(value)
            self.timestamp_buffer.append(current_time)
            
            # Send to queue for external access
            try:
                self.data_queue.put({
                    'timestamp': current_time,
                    'value': value,
                    'label': label
                }, block=False)
            except queue.Full:
                # Remove old data if queue is full
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put({
                        'timestamp': current_time,
                        'value': value,
                        'label': label
                    }, block=False)
                except queue.Empty:
                    pass
            
            sample_count += 1
            
            # Sleep to maintain sampling rate
            time.sleep(sample_interval / 1000)
    
    def start(self):
        """Start the EEG simulation."""
        if not self.is_running:
            self.is_running = True
            self.simulation_thread_obj = threading.Thread(target=self.simulation_thread)
            self.simulation_thread_obj.daemon = True
            self.simulation_thread_obj.start()
            print(f"EEG simulation started at {self.sampling_rate} Hz")
    
    def stop(self):
        """Stop the EEG simulation."""
        self.is_running = False
        if hasattr(self, 'simulation_thread_obj'):
            self.simulation_thread_obj.join(timeout=1)
        print("EEG simulation stopped")
    
    def get_latest_sample(self):
        """Get the latest EEG sample."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_buffer_data(self, num_samples=None):
        """Get recent data from the buffer."""
        if num_samples is None:
            num_samples = len(self.data_buffer)
        
        num_samples = min(num_samples, len(self.data_buffer))
        
        if num_samples == 0:
            return [], []
        
        timestamps = list(self.timestamp_buffer)[-num_samples:]
        values = list(self.data_buffer)[-num_samples:]
        
        return timestamps, values
    
    def trigger_blink(self):
        """Manually trigger a blink (for testing)."""
        if self.current_blink_samples == 0:
            self.current_blink_samples = self.blink_samples_total
            print("Manual blink triggered")
    
    def set_blink_probability(self, probability):
        """Set the probability of automatic blinks."""
        self.blink_probability = max(0, min(1, probability))
        print(f"Blink probability set to {self.blink_probability:.3f}")

if __name__ == "__main__":
    # Test the EEG simulator
    simulator = EEGSimulator(sampling_rate=100)  # Lower rate for testing
    
    try:
        simulator.start()
        
        print("Simulator running... Press Ctrl+C to stop")
        print("Monitoring for 10 seconds...")
        
        blink_count = 0
        sample_count = 0
        
        for i in range(1000):  # Monitor for 10 seconds at 100Hz
            sample = simulator.get_latest_sample()
            if sample:
                sample_count += 1
                if sample['label'] == 'blink':
                    blink_count += 1
                    print(f"BLINK detected at {sample['timestamp']:.1f}ms: {sample['value']:.1f}")
            
            time.sleep(0.01)  # 10ms delay
        
        print(f"\nSimulation complete:")
        print(f"Total samples: {sample_count}")
        print(f"Blinks detected: {blink_count}")
        print(f"Blink rate: {blink_count/10:.2f} blinks/second")
        
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        simulator.stop()
