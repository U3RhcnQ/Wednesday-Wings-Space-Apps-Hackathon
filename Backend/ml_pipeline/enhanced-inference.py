# Enhanced Inference Script for Exoplanet Detection
# NASA Space Apps Challenge 2025
# Load saved models and make predictions with comprehensive metadata support
#
# NOTE: This script requires the following files to be present:
#   - models/BEST_MODEL.joblib (trained model)
#   - metadata/final_scaler.pkl (feature scaler from training)
#   - metadata/stellar_imputer.pkl (stellar feature imputer)
#   - metadata/planetary_imputer.pkl (planetary feature imputer)
#   - metadata/other_imputer.pkl (other feature imputer)
#
# For more comprehensive validation, use real-world-model-test.py

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
from datetime import datetime
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class ExoplanetDetector:
    """
    Enhanced wrapper class for loading and using trained exoplanet detection models
    with comprehensive metadata support and prediction confidence analysis
    """
    
    def __init__(self, model_path='models/BEST_MODEL.joblib'):
        """Load the trained model and all preprocessing objects with metadata"""
        print("🚀 Initializing Enhanced Exoplanet Detector...")
        print("=" * 60)
        
        self.model_path = model_path
        self.initialization_time = datetime.now().isoformat()
        
        # Load model
        try:
            self.model = joblib.load(model_path)
            print(f"   ✅ Model loaded from: {model_path}")
            print(f"   📊 Model type: {type(self.model).__name__}")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise
        
        # Load preprocessing objects (updated paths)
        try:
            self.scaler = joblib.load('metadata/final_scaler.pkl')
            self.stellar_imputer = joblib.load('metadata/stellar_imputer.pkl')
            self.planetary_imputer = joblib.load('metadata/planetary_imputer.pkl')
            self.other_imputer = joblib.load('metadata/other_imputer.pkl')
            
            # Try to load feature names from metadata
            try:
                with open('metadata/feature_metadata.json', 'r') as f:
                    feature_metadata = json.load(f)
                    self.feature_names = feature_metadata.get('feature_names', [])
            except:
                # If no feature metadata, we'll try to infer from scaler
                self.feature_names = []
                print(f"   ⚠️  Feature names not found, will use dynamic feature detection")
            
            print(f"   ✅ Preprocessing objects loaded")
            if self.feature_names:
                print(f"   🔢 Expected features: {len(self.feature_names)}")
        except Exception as e:
            print(f"   ❌ Error loading preprocessing objects: {e}")
            raise
        
        # Load metadata if available
        self.metadata = self._load_model_metadata()
        
        # Initialize prediction history
        self.prediction_history = []
        
        print(f"   ✅ Detector initialized successfully!")
        print("=" * 60)
    
    def _load_model_metadata(self):
        """Load model metadata if available"""
        # Try to find corresponding metadata file
        model_name = os.path.basename(self.model_path).replace('.joblib', '')
        
        # Check for specific model metadata
        metadata_paths = [
            f'metadata/{model_name}_metadata.json',
            'metadata/complete_training_metadata.json',
            'metadata/best_model_metadata.json'
        ]
        
        for path in metadata_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    print(f"   📋 Metadata loaded from: {path}")
                    return metadata
                except Exception as e:
                    print(f"   ⚠️  Could not load metadata from {path}: {e}")
        
        print("   ⚠️  No metadata found - using default configuration")
        return {'model_info': 'No metadata available'}
    
    def preprocess_data(self, X):
        """Preprocess new data using saved preprocessing objects"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if self.feature_names:
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                # Create generic column names
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        # Separate feature types (matching training pipeline)
        feature_cols = list(X_df.columns)
        
        stellar_features = [col for col in feature_cols if 
                           any(x in col for x in ['stellar', 'magnitude'])]
        planetary_features = [col for col in feature_cols if 
                             any(x in col for x in ['planet', 'orbital', 'transit', 
                                                    'equilibrium', 'semi_major', 
                                                    'inclination', 'period', 'duration'])]
        error_features = [col for col in feature_cols if 'err' in col]
        other_features = [col for col in feature_cols if 
                         col not in stellar_features + planetary_features + error_features]
        
        # Apply imputation by feature type
        X_imputed = X_df.copy()
        
        # Stellar features
        if len(stellar_features) > 0:
            stellar_features_valid = [col for col in stellar_features if X_df[col].notna().any()]
            if len(stellar_features_valid) > 0:
                X_imputed[stellar_features_valid] = self.stellar_imputer.transform(X_df[stellar_features_valid])
            all_nan_cols = [col for col in stellar_features if col not in stellar_features_valid]
            if all_nan_cols:
                X_imputed[all_nan_cols] = 0
        
        # Planetary features
        if len(planetary_features) > 0:
            planetary_features_valid = [col for col in planetary_features if X_df[col].notna().any()]
            if len(planetary_features_valid) > 0:
                X_imputed[planetary_features_valid] = self.planetary_imputer.transform(X_df[planetary_features_valid])
            all_nan_cols = [col for col in planetary_features if col not in planetary_features_valid]
            if all_nan_cols:
                X_imputed[all_nan_cols] = 0
        
        # Error features - zero imputation
        if len(error_features) > 0:
            X_imputed[error_features] = X_df[error_features].fillna(0)
        
        # Other features
        if len(other_features) > 0:
            other_features_valid = [col for col in other_features if X_df[col].notna().any()]
            if len(other_features_valid) > 0:
                X_imputed[other_features_valid] = self.other_imputer.transform(X_df[other_features_valid])
            all_nan_cols = [col for col in other_features if col not in other_features_valid]
            if all_nan_cols:
                X_imputed[all_nan_cols] = 0
        
        # Scale features - return as DataFrame to preserve feature names
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        return X_scaled
    
    def predict(self, X, return_proba=False, detailed=False):
        """
        Make predictions on new data with enhanced output options
        
        Parameters:
        - X: Feature matrix (numpy array or pandas DataFrame)
        - return_proba: If True, return probability scores
        - detailed: If True, return detailed prediction information
        
        Returns:
        - predictions: Binary predictions (0=Non-Planet, 1=Planet)
        - probabilities (optional): Probability scores for each class
        - details (optional): Detailed prediction information
        """
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        # Store prediction in history
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': X.shape if hasattr(X, 'shape') else len(X),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        self.prediction_history.append(prediction_record)
        
        if detailed:
            # Calculate confidence metrics
            max_probas = np.max(probabilities, axis=1)
            prediction_confidence = max_probas
            uncertainty = 1 - max_probas
            
            details = {
                'prediction_confidence': prediction_confidence.tolist(),
                'uncertainty_scores': uncertainty.tolist(),
                'high_confidence_threshold': 0.8,
                'high_confidence_count': np.sum(max_probas > 0.8),
                'low_confidence_count': np.sum(max_probas < 0.6),
                'prediction_distribution': dict(Counter(predictions))
            }
            
            if return_proba:
                return predictions, probabilities, details
            else:
                return predictions, details
        
        if return_proba:
            return predictions, probabilities
        
        return predictions
    
    def predict_single(self, features_dict):
        """
        Predict for a single observation from a dictionary of features
        
        Parameters:
        - features_dict: Dictionary with feature names as keys
        
        Returns:
        - Comprehensive prediction result dictionary
        """
        # Validate input
        missing_features = set(self.feature_names) - set(features_dict.keys())
        if missing_features:
            print(f"⚠️  Warning: Missing features will be imputed: {missing_features}")
        
        # Create DataFrame with proper column order
        df = pd.DataFrame([features_dict])
        
        # Ensure all required features are present (fill missing with NaN)
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = np.nan
        
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        # Get prediction with detailed information
        pred, proba, details = self.predict(df.values, return_proba=True, detailed=True)
        
        # Create comprehensive result
        result = {
            'input_summary': {
                'total_features_provided': len(features_dict),
                'total_features_required': len(self.feature_names),
                'missing_features_count': len(missing_features),
                'prediction_timestamp': datetime.now().isoformat()
            },
            'prediction': {
                'class': 'CONFIRMED PLANET' if pred[0] == 1 else 'NON-PLANET',
                'binary_value': int(pred[0]),
                'confidence_level': 'HIGH' if details['prediction_confidence'][0] > 0.8 else 
                                  'MEDIUM' if details['prediction_confidence'][0] > 0.6 else 'LOW'
            },
            'probabilities': {
                'planet_probability': float(proba[0][1]),
                'non_planet_probability': float(proba[0][0]),
                'confidence_score': float(details['prediction_confidence'][0]),
                'uncertainty_score': float(details['uncertainty_scores'][0])
            },
            'model_info': {
                'model_type': type(self.model).__name__,
                'model_path': self.model_path,
                'prediction_method': 'ensemble' if 'Ensemble' in str(type(self.model)) else 'single'
            }
        }
        
        return result
    
    def batch_predict(self, data_path, output_path=None, chunk_size=1000):
        """
        Make predictions on large datasets in batches
        
        Parameters:
        - data_path: Path to CSV file with features
        - output_path: Path to save results (optional)
        - chunk_size: Number of samples to process at once
        """
        print(f"🔄 Processing large dataset: {data_path}")
        
        # Load data in chunks
        try:
            data = pd.read_csv(data_path)
            print(f"   📊 Loaded {len(data):,} samples")
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return None
        
        # Ensure required features are present
        available_features = [f for f in self.feature_names if f in data.columns]
        print(f"   🔢 Available features: {len(available_features)}/{len(self.feature_names)}")
        
        # Process in chunks
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            chunk_features = chunk[available_features].values
            
            pred, proba = self.predict(chunk_features, return_proba=True)
            
            all_predictions.extend(pred)
            all_probabilities.extend(proba[:, 1])  # Planet probabilities
            
            print(f"   ⏳ Processed {min(i+chunk_size, len(data)):,}/{len(data):,} samples")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'prediction': all_predictions,
            'planet_probability': all_probabilities,
            'prediction_class': ['PLANET' if p == 1 else 'NON-PLANET' for p in all_predictions],
            'confidence_level': ['HIGH' if p > 0.8 else 'MEDIUM' if p > 0.6 else 'LOW' 
                                for p in all_probabilities]
        })
        
        # Add summary statistics
        summary = {
            'total_samples': len(results_df),
            'predicted_planets': (results_df['prediction'] == 1).sum(),
            'predicted_non_planets': (results_df['prediction'] == 0).sum(),
            'planet_percentage': ((results_df['prediction'] == 1).sum() / len(results_df)) * 100,
            'high_confidence_predictions': (results_df['confidence_level'] == 'HIGH').sum(),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        print(f"   📊 Batch Prediction Summary:")
        print(f"      - Predicted Planets: {summary['predicted_planets']:,} ({summary['planet_percentage']:.2f}%)")
        print(f"      - High Confidence: {summary['high_confidence_predictions']:,}")
        
        # Save results if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            
            # Save summary
            summary_path = output_path.replace('.csv', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            print(f"   💾 Results saved: {output_path}")
            print(f"   📋 Summary saved: {summary_path}")
        
        return results_df, summary
    
    def get_prediction_statistics(self):
        """Get statistics about predictions made during this session"""
        if not self.prediction_history:
            return {"message": "No predictions made yet"}
        
        total_predictions = sum(len(record['predictions']) for record in self.prediction_history)
        total_planets = sum(sum(record['predictions']) for record in self.prediction_history)
        
        stats = {
            'session_info': {
                'detector_initialized': self.initialization_time,
                'total_prediction_calls': len(self.prediction_history),
                'total_samples_predicted': total_predictions
            },
            'prediction_summary': {
                'total_planet_predictions': total_planets,
                'total_non_planet_predictions': total_predictions - total_planets,
                'planet_percentage': (total_planets / total_predictions * 100) if total_predictions > 0 else 0
            },
            'model_metadata': self.metadata.get('model_info', {}) if isinstance(self.metadata, dict) else {}
        }
        
        return stats
    
    def create_prediction_confidence_plot(self, probabilities, save_path='plots/prediction_confidence.png'):
        """Create visualization of prediction confidence distribution"""
        
        plt.figure(figsize=(12, 8))
        
        # Extract planet probabilities
        if len(probabilities.shape) == 2:
            planet_probs = probabilities[:, 1]
        else:
            planet_probs = probabilities
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confidence Distribution
        ax1.hist(planet_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', label='Decision Boundary')
        ax1.axvline(0.8, color='green', linestyle='--', label='High Confidence')
        ax1.axvline(0.2, color='orange', linestyle='--', label='High Confidence (Non-Planet)')
        ax1.set_xlabel('Planet Probability')
        ax1.set_ylabel('Count')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence Categories
        high_conf_planet = np.sum(planet_probs > 0.8)
        medium_conf_planet = np.sum((planet_probs > 0.6) & (planet_probs <= 0.8))
        low_conf = np.sum((planet_probs >= 0.4) & (planet_probs <= 0.6))
        medium_conf_non_planet = np.sum((planet_probs >= 0.2) & (planet_probs < 0.4))
        high_conf_non_planet = np.sum(planet_probs < 0.2)
        
        categories = ['High\\nPlanet', 'Medium\\nPlanet', 'Uncertain', 'Medium\\nNon-Planet', 'High\\nNon-Planet']
        counts = [high_conf_planet, medium_conf_planet, low_conf, medium_conf_non_planet, high_conf_non_planet]
        colors = ['darkgreen', 'lightgreen', 'yellow', 'orange', 'red']
        
        ax2.bar(categories, counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Confidence Categories')
        ax2.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(counts):
            ax2.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 3. Cumulative Distribution
        sorted_probs = np.sort(planet_probs)
        y_vals = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        ax3.plot(sorted_probs, y_vals, linewidth=2)
        ax3.axvline(0.5, color='red', linestyle='--', label='Decision Boundary')
        ax3.set_xlabel('Planet Probability')
        ax3.set_ylabel('Cumulative Proportion')
        ax3.set_title('Cumulative Distribution of Planet Probabilities')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box Plot by Prediction
        predictions = (planet_probs > 0.5).astype(int)
        planet_pred_probs = planet_probs[predictions == 1]
        non_planet_pred_probs = planet_probs[predictions == 0]
        
        box_data = [non_planet_pred_probs, planet_pred_probs]
        box_labels = ['Predicted\\nNon-Planet', 'Predicted\\nPlanet']
        
        ax4.boxplot(box_data, labels=box_labels, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax4.axhline(0.5, color='red', linestyle='--', label='Decision Boundary')
        ax4.set_ylabel('Planet Probability')
        ax4.set_title('Probability Distribution by Prediction')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Ensure plots directory exists
        os.makedirs('plots', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Confidence plot saved: {save_path}")

def main():
    """Example usage of the Enhanced Exoplanet Detector"""
    print("=" * 80)
    print("ENHANCED EXOPLANET DETECTOR - INFERENCE EXAMPLE")
    print("=" * 80)
    
    try:
        # Initialize detector
        detector = ExoplanetDetector()
        
        # Example 1: Single prediction
        print(f"\\n🔮 Example 1: Single Prediction")
        print("-" * 40)
        
        # Create example feature dictionary (you would use real exoplanet data)
        example_features = {
            'koi_period': 365.25,  # Earth-like orbital period
            'koi_impact': 0.2,     # Impact parameter
            'koi_duration': 3.5,   # Transit duration
            'koi_depth': 1000,     # Transit depth in ppm
            'koi_prad': 1.0,       # Planet radius relative to Earth
            'koi_teq': 288,        # Equilibrium temperature
            # Add more features as needed based on your dataset
        }
        
        result = detector.predict_single(example_features)
        
        print(f"Prediction: {result['prediction']['class']}")
        print(f"Planet Probability: {result['probabilities']['planet_probability']:.4f}")
        print(f"Confidence: {result['prediction']['confidence_level']}")
        
        # Example 2: Load and predict on test data
        print(f"\\n🔮 Example 2: Batch Prediction")
        print("-" * 40)
        
        test_data_path = 'data/features_processed.csv'
        
        if os.path.exists(test_data_path):
            # Load sample of test data
            test_data = pd.read_csv(test_data_path).head(100)  # First 100 samples
            
            predictions, probabilities, details = detector.predict(
                test_data.values, return_proba=True, detailed=True
            )
            
            print(f"Batch Prediction Results:")
            print(f"   - Total samples: {len(predictions)}")
            print(f"   - Predicted planets: {sum(predictions)}")
            print(f"   - High confidence predictions: {details['high_confidence_count']}")
            print(f"   - Average planet probability: {np.mean(probabilities[:, 1]):.4f}")
            
            # Create confidence visualization
            detector.create_prediction_confidence_plot(probabilities)
            
        else:
            print(f"   ⚠️  Test data not found at {test_data_path}")
            print("   Please run the preprocessing pipeline first")
        
        # Example 3: Session statistics
        print(f"\\n📊 Session Statistics")
        print("-" * 40)
        
        stats = detector.get_prediction_statistics()
        print(f"Prediction calls made: {stats['session_info']['total_prediction_calls']}")
        print(f"Total samples predicted: {stats['session_info']['total_samples_predicted']}")
        
        print(f"\\n✅ Inference examples completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Inference failed: {str(e)}")
        print("\\nPlease ensure you have:")
        print("   1. Trained model: models/BEST_MODEL.joblib")
        print("   2. Scaler: metadata/final_scaler.pkl")
        print("   3. Imputers: metadata/stellar_imputer.pkl")
        print("                metadata/planetary_imputer.pkl")
        print("                metadata/other_imputer.pkl")
        print("\\n💡 Tip: Run the training pipeline first to generate these files")
        print("   Or use real-world-model-test.py for comprehensive validation")
        raise

if __name__ == "__main__":
    main()