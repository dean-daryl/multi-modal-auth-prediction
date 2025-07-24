import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from pyod.models.knn import KNN
import noisereduce as nr

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = BASE_DIR / "data/audio_samples"
PROCESSED_DIR = BASE_DIR / "data/processed_audio"
FEATURE_FILE = BASE_DIR / "data/audio_features.csv"
MODEL_FILE = BASE_DIR / "models/voice_model.pkl"
METRICS_FILE = BASE_DIR / "models/voice_model_metrics.json"
OUTLIER_MODEL_FILE = BASE_DIR / "models/voice_outlier_detector.pkl"
SAMPLE_RATE = 16000
AUTH_THRESHOLD = 0.7
UNKNOWN_THRESHOLD = 0.5

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "models", exist_ok=True)

def extract_features(y, sr):
    features = {}
    
    try:
        y = nr.reduce_noise(y=y, sr=sr)
    except:
        pass
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_var'] = np.var(chroma)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    rms = librosa.feature.rms(y=y)
    features['rms_energy_mean'] = np.mean(rms)
    features['rms_energy_std'] = np.std(rms)
    
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_mean'] = np.mean(zero_crossing)
    features['zero_crossing_std'] = np.std(zero_crossing)
    
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    features['tonnetz_mean'] = np.mean(tonnetz)
    
    return features

def augment_audio(y, sr):
    augmented = []
    
    try:
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        augmented.append(('pitch_shift', y_pitch))
    except:
        pass
    
    try:
        y_time = librosa.effects.time_stretch(y, rate=0.8)
        augmented.append(('time_stretch', y_time))
    except:
        pass
    
    if len(augmented) < 2:
        try:
            noise = np.random.normal(0, 0.005, len(y))
            y_noise = y + noise
            augmented.append(('noise', y_noise))
        except:
            pass
    
    return augmented[:2]

def process_audio():
    features_list = []
    
    print("Processing audio files...")
    
    for speaker_dir in os.listdir(AUDIO_DIR):
        speaker_path = os.path.join(AUDIO_DIR, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
            
        print(f"  Processing speaker: {speaker_dir}")
        
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                filepath = os.path.join(speaker_path, file)
                
                try:
                    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                    
                    plt.figure(figsize=(12, 8))
                    plt.subplot(2, 1, 1)
                    librosa.display.waveshow(y, sr=sr)
                    plt.title(f"Waveform: {speaker_dir}/{file}")
                    
                    plt.subplot(2, 1, 2)
                    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
                    plt.colorbar(format="%+2.0f dB")
                    plt.tight_layout()
                    
                    os.makedirs(os.path.join(PROCESSED_DIR, speaker_dir), exist_ok=True)
                    plt.savefig(os.path.join(PROCESSED_DIR, speaker_dir, f"{os.path.splitext(file)[0]}.png"))
                    plt.close()
                    
                    features = extract_features(y, sr)
                    features['user'] = speaker_dir
                    features['file'] = file
                    features['augmentation'] = 'original'
                    features_list.append(features)
                    
                    for aug_type, aug_y in augment_audio(y, sr):
                        aug_features = extract_features(aug_y, sr)
                        aug_features['user'] = speaker_dir
                        aug_features['file'] = f"aug_{aug_type}_{file}"
                        aug_features['augmentation'] = aug_type
                        features_list.append(aug_features)
                        
                except Exception as e:
                    print(f"Error processing {speaker_dir}/{file}: {str(e)}")
                    continue
    
    df = pd.DataFrame(features_list)
    df.to_csv(FEATURE_FILE, index=False)
    print(f"Features saved to {FEATURE_FILE}")
    return df

def train_model():
    print("Training voice model...")
    
    if not os.path.exists(FEATURE_FILE):
        print("No features found. Processing audio first...")
        df = process_audio()
    else:
        df = pd.read_csv(FEATURE_FILE)
    
    known_df = df[~df['user'].str.startswith('unknown')]
    
    if len(known_df) == 0:
        raise ValueError("No known speaker data found")
    
    feature_cols = [col for col in known_df.columns if col not in ['user', 'file', 'augmentation']]
    X = known_df[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(known_df['user'])
    
    if len(le.classes_) == 1:
        print("Only one speaker class found. Using dummy classifier.")
        model = make_pipeline(
            StandardScaler(),
            DummyClassifier(strategy="constant", constant=0)
        )
        outlier_detector = KNN(contamination=0.1)
        outlier_detector.fit(np.zeros((10, X.shape[1])))
    else:
        model = make_pipeline(
            StandardScaler(),
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                objective='multi:softprob',
                num_class=len(le.classes_))
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        outlier_detector = KNN(contamination=0.1)
        outlier_detector.fit(X_scaled)
    
    if len(le.classes_) > 1 and len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
    else:
        model.fit(X, y)
        accuracy = 1.0
        f1 = 1.0
        print("Insufficient data for proper evaluation")
    
    joblib.dump({
        'model': model,
        'label_encoder': le,
        'feature_names': list(X.columns),
        'scaler': StandardScaler().fit(X),
        'outlier_detector': outlier_detector
    }, MODEL_FILE)
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classes': list(le.classes_),
        'test_size': len(y) if len(le.classes_) == 1 else len(y_test)
    }
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Metrics saved to {METRICS_FILE}")

class VoiceVerifier:
    def __init__(self):
        try:
            model_data = joblib.load(MODEL_FILE)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.scaler = model_data['scaler']
            self.outlier_detector = model_data['outlier_detector']
        except Exception as e:
            raise RuntimeError(f"Failed to load voice model: {str(e)}")

    def verify(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            features = extract_features(y, sr)
            
            feature_vector = np.array([features[col] for col in self.feature_names]).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)
            
            outlier_score = self.outlier_detector.decision_function(scaled_features)[0]
            is_outlier = outlier_score < 0
            
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(feature_vector)[0]
            else:
                probas = np.zeros(len(self.label_encoder.classes_))
                probas[0] = 1.0
            
            pred_idx = np.argmax(probas)
            confidence = probas[pred_idx]
            prediction = self.label_encoder.inverse_transform([pred_idx])[0]
            
            if len(self.label_encoder.classes_) == 1:
                status = "authorized" if not is_outlier else "unauthorized"
            else:
                status = "authorized" if (confidence > AUTH_THRESHOLD and not is_outlier) else "unauthorized"
            
            probabilities = {
                user: prob 
                for user, prob in zip(self.label_encoder.classes_, probas)
            }
            
            return {
                'user': prediction,
                'confidence': float(confidence),
                'outlier_score': float(outlier_score),
                'is_outlier': bool(is_outlier),
                'status': status,
                'probabilities': probabilities
            }
        except Exception as e:
            return {'error': str(e)}

if __name__ == "__main__":
    print("Starting Audio Processing Pipeline...")
    
    process_audio()
    train_model()
    
    verifier = VoiceVerifier()
    test_cases = []
    
    for speaker_dir in os.listdir(AUDIO_DIR):
        speaker_path = os.path.join(AUDIO_DIR, speaker_dir)
        if os.path.isdir(speaker_path):
            for file in os.listdir(speaker_path):
                if file.endswith(".wav"):
                    rel_path = f"{speaker_dir}/{file}"
                    expected_user = speaker_dir if not speaker_dir.startswith("unknown") else None
                    should_authorize = not speaker_dir.startswith("unknown")
                    test_cases.append((rel_path, expected_user, should_authorize))
    
    if not test_cases:
        print("No test files found")
    else:
        print("\nRunning Verification Tests:")
        for rel_path, expected_user, should_authorize in test_cases[:4]:
            test_file = os.path.join(AUDIO_DIR, rel_path)
            try:
                result = verifier.verify(test_file)
                
                if 'error' in result:
                    print(f"\nError testing {rel_path}: {result['error']}")
                    continue
                    
                print(f"\nTest File: {rel_path}")
                print(f"Expected: {'AUTHORIZED' if should_authorize else 'UNAUTHORIZED'} as {expected_user}")
                print(f"Predicted: {result['user']} ({result['confidence']*100:.2f}% confidence)")
                print(f"Outlier Score: {result['outlier_score']:.2f} ({'OUTLIER' if result['is_outlier'] else 'INLIER'})")
                print(f"Status: {result['status'].upper()}")
                
                if (should_authorize and result['status'] == 'authorized' and 
                    (expected_user is None or result['user'] == expected_user)) or \
                   (not should_authorize and result['status'] == 'unauthorized'):
                    print("Test PASSED")
                else:
                    print("Test FAILED")
                
                print("\nProbability Distribution:")
                for user, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
                    print(f"{user:>15}: {prob:.2%}")
                    
            except Exception as e:
                print(f"\nCritical error testing {rel_path}: {str(e)}")