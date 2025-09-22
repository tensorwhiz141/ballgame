"""
Data processing module for EEG blink detection.
Combines datasets, preprocesses data, and extracts features for ML training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EEGDataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.combined_data = None
        self.features = None
        self.labels = None
        
    def load_datasets(self):
        """Load and combine the two EEG datasets."""
        print("Loading datasets...")
        
        # Load dataset 1
        dataset1_path = os.path.join(self.data_dir, "exg_blink_dataset1 (1).csv")
        df1 = pd.read_csv(dataset1_path)
        df1['dataset'] = 'dataset1'
        
        # Load dataset 2
        dataset2_path = os.path.join(self.data_dir, "exg_blink_dataset2000.csv")
        df2 = pd.read_csv(dataset2_path)
        df2['dataset'] = 'dataset2'
        
        # Combine datasets
        self.combined_data = pd.concat([df1, df2], ignore_index=True)
        
        print(f"Dataset 1 shape: {df1.shape}")
        print(f"Dataset 2 shape: {df2.shape}")
        print(f"Combined dataset shape: {self.combined_data.shape}")
        
        # Display basic statistics
        print("\nLabel distribution:")
        print(self.combined_data['label'].value_counts())
        
        return self.combined_data
    
    def clean_data(self):
        """Clean the combined dataset."""
        print("\nCleaning data...")
        
        # Check for missing values
        missing_values = self.combined_data.isnull().sum()
        print(f"Missing values:\n{missing_values}")
        
        # Remove any rows with missing values
        self.combined_data = self.combined_data.dropna()
        
        # Check for duplicates
        duplicates = self.combined_data.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Remove duplicates
        self.combined_data = self.combined_data.drop_duplicates()
        
        # Sort by timestamp
        self.combined_data = self.combined_data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Data shape after cleaning: {self.combined_data.shape}")
        
        return self.combined_data
    
    def extract_features(self, window_size=10):
        """Extract features from EEG signals for ML training."""
        print(f"\nExtracting features with window size: {window_size}")
        
        features_list = []
        labels_list = []
        
        # Group by dataset to maintain temporal order
        for dataset_name in self.combined_data['dataset'].unique():
            dataset_subset = self.combined_data[self.combined_data['dataset'] == dataset_name].copy()
            
            # Create sliding windows
            for i in range(len(dataset_subset) - window_size + 1):
                window = dataset_subset.iloc[i:i+window_size]
                
                # Extract statistical features from the window
                features = {
                    'mean': window['value'].mean(),
                    'std': window['value'].std(),
                    'min': window['value'].min(),
                    'max': window['value'].max(),
                    'median': window['value'].median(),
                    'range': window['value'].max() - window['value'].min(),
                    'skewness': window['value'].skew(),
                    'kurtosis': window['value'].kurtosis(),
                    'rms': np.sqrt(np.mean(window['value']**2)),
                    'variance': window['value'].var(),
                }
                
                # Add time-domain features
                values = window['value'].values
                features['slope'] = np.polyfit(range(len(values)), values, 1)[0]
                features['energy'] = np.sum(values**2)
                
                # Add frequency domain features (simple)
                fft_values = np.fft.fft(values)
                features['dominant_freq'] = np.argmax(np.abs(fft_values))
                features['spectral_energy'] = np.sum(np.abs(fft_values)**2)
                
                features_list.append(features)
                
                # Label is determined by the center point of the window
                center_idx = i + window_size // 2
                labels_list.append(dataset_subset.iloc[center_idx]['label'])
        
        # Convert to DataFrame
        self.features = pd.DataFrame(features_list)
        self.labels = pd.Series(labels_list)
        
        print(f"Features shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Feature columns: {list(self.features.columns)}")
        
        return self.features, self.labels
    
    def normalize_features(self):
        """Normalize features using StandardScaler."""
        print("\nNormalizing features...")
        
        self.features_normalized = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=self.features.columns
        )
        
        print("Features normalized successfully.")
        return self.features_normalized
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print(f"\nSplitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features_normalized, 
            self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training labels distribution:\n{y_train.value_counts()}")
        print(f"Test labels distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def visualize_data(self):
        """Create visualizations of the data."""
        print("\nCreating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Raw signal comparison
        normal_data = self.combined_data[self.combined_data['label'] == 'normal']['value']
        blink_data = self.combined_data[self.combined_data['label'] == 'blink']['value']
        
        axes[0, 0].hist(normal_data, alpha=0.7, label='Normal', bins=50)
        axes[0, 0].hist(blink_data, alpha=0.7, label='Blink', bins=50)
        axes[0, 0].set_title('EEG Value Distribution')
        axes[0, 0].set_xlabel('EEG Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Plot 2: Feature correlation heatmap
        correlation_matrix = self.features.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlation Matrix')
        
        # Plot 3: Feature importance (mean values by class)
        feature_means = self.features.groupby(self.labels).mean()
        feature_means.T.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Means by Class')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].legend(['Blink', 'Normal'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Sample time series
        sample_data = self.combined_data.head(1000)
        axes[1, 1].plot(sample_data['timestamp'], sample_data['value'])
        blink_points = sample_data[sample_data['label'] == 'blink']
        axes[1, 1].scatter(blink_points['timestamp'], blink_points['value'], 
                          color='red', s=20, alpha=0.7, label='Blinks')
        axes[1, 1].set_title('Sample EEG Signal with Blinks')
        axes[1, 1].set_xlabel('Timestamp')
        axes[1, 1].set_ylabel('EEG Value')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'visualizations.png'")
    
    def process_all(self, window_size=10, test_size=0.2):
        """Run the complete data processing pipeline."""
        print("Starting complete data processing pipeline...")
        
        # Load and combine datasets
        self.load_datasets()
        
        # Clean data
        self.clean_data()
        
        # Extract features
        self.extract_features(window_size=window_size)
        
        # Normalize features
        self.normalize_features()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(test_size=test_size)
        
        # Create visualizations (skip for now to avoid GUI issues)
        # self.visualize_data()
        
        print("\nData processing pipeline completed successfully!")
        
        return X_train, X_test, y_train, y_test, self.scaler

if __name__ == "__main__":
    # Example usage
    processor = EEGDataProcessor(data_dir="data")
    X_train, X_test, y_train, y_test, scaler = processor.process_all()
