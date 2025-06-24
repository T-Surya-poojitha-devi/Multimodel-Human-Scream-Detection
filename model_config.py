# model_config.py

MODEL_CONFIG = {
    'input_shape': (40, 862, 1),  # (mfcc features, time steps, channels)
    'n_classes': 6,
    'emotion_labels': ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprised']
}