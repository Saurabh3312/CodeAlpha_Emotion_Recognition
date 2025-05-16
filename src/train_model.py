import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report

# Set data path
DATA_PATH = "../data/"

# Extract features from an audio file
def extract_features(file_name):
    try:
        audio_data, sample_rate = sf.read(file_name)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

# Prepare dataset
def prepare_dataset():
    features = []
    labels = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".wav"):
            file_path = os.path.join(DATA_PATH, file)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(file.split("_")[0])  # Assuming filename format: emotion_filename.wav
    return np.array(features), np.array(labels)

# Load data
X, y = prepare_dataset()

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
predictions = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
print("\n📊 Classification Report:")
print(classification_report(y_test_labels, predictions))

# Save the trained model
model.save("../models/emotion_recognition_model.h5")
