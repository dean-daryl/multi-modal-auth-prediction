import joblib
import numpy as np
import sys
import cv2
import face_recognition
import librosa
import os
import noisereduce as nr
import pandas as pd


# Paths to models (adjust if needed)
FACE_MODEL_PATH = "../models/face_model.pkl"
PCA_MODEL_PATH = "../models/pca_model.pkl"
VOICE_MODEL_PATH = "../models/voice_model.pkl"
PRODUCT_MODEL_PATH = "../models/product_recommendation_model.pkl"
MERGED_DATA_PATH = "../data/merged_data.csv"

# Hard-coded feature columns for product recommendation
PRODUCT_FEATURE_COLS = [
    'engagement_score', 'purchase_interest_score', 'customer_rating', 'purchase_amount',
    'review_sentiment_Negative', 'review_sentiment_Neutral', 'review_sentiment_Positive', 'review_sentiment_Unknown',
    'social_media_platform_Facebook', 'social_media_platform_Instagram', 'social_media_platform_LinkedIn',
    'social_media_platform_TikTok', 'social_media_platform_Twitter', 'social_media_platform_Unknown'
]

AUTH_THRESHOLD = 0.6
SAMPLE_RATE = 16000

# --- Face Feature Extraction ---
def extract_face(image_path):
    """
    Loads an image, detects the largest face using face_recognition, crops, resizes to 100x100, converts to grayscale, and flattens.
    Returns None if no face is found.
    """
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    if not locations:
        print("No face detected in image.")
        return None
    # Use the first detected face
    top, right, bottom, left = locations[0]
    face = image[top:bottom, left:right]
    # Resize to 100x100
    face_resized = cv2.resize(face, (100, 100))
    # Convert to grayscale
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
    return face_gray.flatten()

# --- Voice Feature Extraction ---
def extract_voice_features(audio_path):
    """
    Loads an audio file and extracts features compatible with the trained model.
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    try:
        y = nr.reduce_noise(y=y, sr=sr)
    except Exception:
        pass

    features = {}
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

# --- Face Recognition ---
def predict_face(image_path, face_model, pca):
    face = extract_face(image_path)
    if face is None:
        return {"status": "unauthorized", "reason": "No face found"}

    print("Extracted face shape:", face.shape)  # Should be (10000,) if 100x100 grayscale
    print("Expected PCA input size (pca.n_features_):", pca.n_features_in_)

    face_pca = pca.transform([face.flatten()])
    print("PCA-transformed shape:", face_pca.shape)

    pred = face_model.predict(face_pca)[0]
    confidence = face_model.predict_proba(face_pca).max()
    status = "authorized" if confidence > AUTH_THRESHOLD else "unauthorized"
    return {"status": status, "user": pred, "confidence": confidence}

# --- Voice Recognition ---
def verify_voice(audio_path, voice_model_data):
    features = extract_voice_features(audio_path)
    feature_names = voice_model_data['feature_names']
    scaler = voice_model_data['scaler']
    outlier_detector = voice_model_data['outlier_detector']
    model = voice_model_data['model']
    label_encoder = voice_model_data['label_encoder']

    feature_vector = np.array([features[col] for col in feature_names]).reshape(1, -1)
    scaled_features = scaler.transform(feature_vector)
    outlier_score = outlier_detector.decision_function(scaled_features)[0]
    is_outlier = outlier_score < 0
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(feature_vector)[0]
    else:
        probas = np.zeros(len(label_encoder.classes_))
        probas[0] = 1.0
    pred_idx = np.argmax(probas)
    confidence = probas[pred_idx]
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    status = "authorized" if (confidence > AUTH_THRESHOLD and not is_outlier) else "unauthorized"
    return {
        "status": status,
        "user": prediction,
        "confidence": confidence,
        "outlier_score": outlier_score,
        "is_outlier": is_outlier
    }

def main():
    print("=== Multimodal Auth Demo ===")
    # Load models
    try:
        face_model = joblib.load(FACE_MODEL_PATH)
        pca = joblib.load(PCA_MODEL_PATH)
        voice_model_data = joblib.load(VOICE_MODEL_PATH)
        product_model = joblib.load(PRODUCT_MODEL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    face_path = input("Enter path to face image: ").strip()
    face_result = predict_face(face_path, face_model, pca)
    print(f"Face Recognition: {face_result}")
    if face_result["status"] != "authorized":
        print("❌ Unauthorized face attempt. Exiting.")
        sys.exit(1)

    audio_path = input("Enter path to voice sample: ").strip()
    voice_result = verify_voice(audio_path, voice_model_data)
    print(f"Voice Recognition: {voice_result}")
    if voice_result["status"] != "authorized":
        print("❌ Unauthorized voice attempt. Exiting.")
        sys.exit(1)

    # --- Product Recommendation ---
    # Default to user_id = '100' if not provided
    try:
        df = pd.read_csv(MERGED_DATA_PATH)
        # Ensure user_id is the same type as in the CSV
        # Assume user_id is an integer and is 100
        print(df['customer_id_new'].unique())
        user_rows = df[df['customer_id_new'] == '100']
        if user_rows.empty:
            print("No data found for user. Cannot generate recommendation.")
            sys.exit(1)
        user_features = user_rows.iloc[-1]
        X_user = user_features[PRODUCT_FEATURE_COLS].values.reshape(1, -1)
        predicted_category = product_model.predict(X_user)[0]
        print(f"🎁 Recommended Product Category: {predicted_category}")
    except Exception as e:
        print(f"Error generating product recommendation: {e}")
        sys.exit(1)

    print(f"✅ Authorized user: {face_result['user']} (Face), {voice_result['user']} (Voice)")
    print("Transaction Approved!")

if __name__ == "__main__":
    main()
