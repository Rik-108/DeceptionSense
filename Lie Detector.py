import os
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from moviepy.editor import VideoFileClip

# -----------------------
# Step 1: Extract audio from videos
# -----------------------
def extract_audio_from_video(video_path, output_path, sr=16000):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path, fps=sr)

# -----------------------
# Step 2: Feature Extraction
# -----------------------
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    inputs = processor(y, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return np.concatenate([mfcc, emb])

# -----------------------
# Step 3: Load Dataset
# -----------------------
def load_dataset(truth_dir, lie_dir, temp_audio_dir="audio_clips"):
    os.makedirs(temp_audio_dir, exist_ok=True)
    X, y = [], []

    # Truthful samples
    for file in os.listdir(truth_dir):
        if file.endswith(".mp4"):
            video_path = os.path.join(truth_dir, file)
            audio_path = os.path.join(temp_audio_dir, f"{file}_truth.wav")
            extract_audio_from_video(video_path, audio_path)
            features = extract_features(audio_path)
            X.append(features)
            y.append(1)

    # Deceptive samples
    for file in os.listdir(lie_dir):
        if file.endswith(".mp4"):
            video_path = os.path.join(lie_dir, file)
            audio_path = os.path.join(temp_audio_dir, f"{file}_lie.wav")
            extract_audio_from_video(video_path, audio_path)
            features = extract_features(audio_path)
            X.append(features)
            y.append(0)

    return np.array(X), np.array(y)

# -----------------------
# Step 4: Real-time mic input
# -----------------------
def record_audio(filename="audio1.wav", duration=5, sr=16000):
    print("🎙️ Speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(filename, audio, sr)

# -----------------------
# Load Pretrained Wav2Vec2
# -----------------------
print("Loading Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# -----------------------
# Step 5: Train the Classifier
# -----------------------
print("Extracting features from dataset...")
X, y = load_dataset("truthful", "deceptive")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc * 100:.2f}%")

# -----------------------
# Step 6: Live Test
# -----------------------
record_audio("live_test.wav", duration=5)
test_feat = extract_features("live_test.wav")
pred = clf.predict([test_feat])[0]
prob = clf.predict_proba([test_feat])[0]
print(f"🎯 Live Prediction: {'Truth' if pred == 1 else 'Lie'} | Confidence: {max(prob) * 100:.2f}%")
