import librosa
import os
import numpy as np
import json
import warnings

def extract_features(directory, json_path, frame_size=2048, hop_length=126, num_frames=20):
    features = {
        "genres_labels": [],
        "mel_spectrograms": [],
        "targets": []
    }

    warnings.filterwarnings("ignore")

    for genre_code, genre_dir in enumerate(os.scandir(directory)):
        if genre_dir.is_dir():
            genre_label = genre_dir.name
            features["genres_labels"].append(genre_label)
            
            print(f"Processing files in the {genre_label} genre")

            for audio_file in os.scandir(genre_dir.path):
                if audio_file.is_file():
                    audio_data, sample_rate = librosa.load(audio_file.path)

                    for i in range(0, min(len(audio_data) - frame_size + 1, num_frames * hop_length), hop_length):
                        frame = audio_data[i: i + frame_size]
                        
                        mel_spectrogram = librosa.feature.melspectrogram(y=frame, sr=sample_rate)
                        features["mel_spectrograms"].append(mel_spectrogram.tolist())
                        
                        target = np.zeros((10), dtype=np.float32)
                        target[genre_code] = 1.0
                        features["targets"].append(target.tolist())
                        
    feat_array = np.array(features["mel_spectrograms"])
    target_array = np.array(features["targets"])
    print("Mel spectrogram array shape:", feat_array.shape)
    print("Target array shape:", target_array.shape)

    with open(json_path, "w") as json_file:
        json.dump(features, json_file, indent=4)

    print("Feature extraction completed!")

extract_features('genres/', 'extracted_features.json')
