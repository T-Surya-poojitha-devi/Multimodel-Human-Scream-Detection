from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
import os
import base64
import logging
import sys
import audioread
from model_config import MODEL_CONFIG
from audio_processor import extract_features, AudioFeatureExtractor
from scream_detector import ScreamDetector
from sentiment_analyzer import SentimentAnalyzer


# At the top of app.py:
CURRENT_USER = "FaheemKhan0817"
TIMESTAMP_UTC = "2025-01-30 05:31:44"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'app_{CURRENT_USER}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_PAD_LEN = MODEL_CONFIG['input_shape'][1]
EMOTION_LABELS = MODEL_CONFIG['emotion_labels']
MAX_AUDIO_LENGTH = 30  # Maximum audio length in seconds
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global components
loaded_model = None
scream_detector = None
sentiment_analyzer = None
audio_extractor = None

def normalize_emotion(emotion):
    """Normalize emotion string to lowercase"""
    if emotion is None:
        return 'unknown'
    return str(emotion).lower().strip()

def load_emotion_model():
    """Load and validate the emotion detection model"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'converted_model')
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return None
            
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        dummy_input = np.zeros((1,) + MODEL_CONFIG['input_shape'])
        test_output = model.predict(dummy_input, verbose=0)
        
        if test_output.shape[1] != MODEL_CONFIG['n_classes']:
            logger.error(f"Model output shape mismatch")
            return None
            
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def initialize_components():
    """Initialize all required components"""
    global loaded_model, scream_detector, sentiment_analyzer, audio_extractor
    
    try:
        loaded_model = load_emotion_model()
        scream_detector = ScreamDetector()
        sentiment_analyzer = SentimentAnalyzer()
        audio_extractor = AudioFeatureExtractor()
        
        status = {
            'model': loaded_model is not None,
            'scream_detector': scream_detector is not None,
            'sentiment_analyzer': sentiment_analyzer is not None,
            'audio_extractor': audio_extractor is not None
        }
        
        if not all(status.values()):
            failed_components = [k for k, v in status.items() if not v]
            logger.error(f"Failed to initialize components: {failed_components}")
            return False
            
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return False

# Initialize components at startup
if not initialize_components():
    logger.error("Application startup failed due to initialization errors")

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint for audio analysis"""
    start_time = datetime.now(timezone.utc)
    logger.info(f"New prediction request received from {CURRENT_USER}")
    
    if not all([loaded_model, scream_detector, sentiment_analyzer, audio_extractor]):
        return jsonify({
            'status': 'error',
            'message': 'One or more components not initialized',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 503

    try:
        # Validate request
        if not request.json or 'audio' not in request.json:
            return jsonify({
                'status': 'error',
                'message': 'No audio data provided'
            }), 400
            
        audio_data = request.json['audio']
        
        # Process base64 audio
        try:
            audio_parts = audio_data.split(',')
            audio_str = audio_parts[1] if len(audio_parts) > 1 else audio_parts[0]
            audio_binary = base64.b64decode(audio_str)
            
            if len(audio_binary) > MAX_FILE_SIZE:
                return jsonify({
                    'status': 'error',
                    'message': f'Audio file too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB'
                }), 400
                
        except Exception as e:
            logger.error(f"Audio decoding failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid audio encoding'
            }), 400

        # Process audio file
        temp_path = None
        try:
            # Save temporary file
            os.makedirs('uploads', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_filename = f'audio_{CURRENT_USER}_{timestamp}_{os.urandom(4).hex()}.wav'
            temp_path = os.path.join('uploads', temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(audio_binary)
            
            try:
                # Load and validate audio
                audio_data, sample_rate = librosa.load(temp_path, sr=None)
            except audioread.exceptions.NoBackendError:
                logger.error("No audio backend available. Please install ffmpeg.")
                return jsonify({
                    'status': 'error',
                    'message': 'Audio processing backend not available. Please ensure ffmpeg is installed.'
                }), 500
            except Exception as e:
                logger.error(f"Error loading audio: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'Error loading audio: {str(e)}'
                }), 500

            # Check for valid audio data
            if len(audio_data) == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'Empty audio file'
                }), 400
            
            duration = len(audio_data) / sample_rate
            if duration > MAX_AUDIO_LENGTH:
                return jsonify({
                    'status': 'error',
                    'message': f'Audio too long. Maximum duration: {MAX_AUDIO_LENGTH} seconds'
                }), 400
            
            if np.max(np.abs(audio_data)) < 0.01:
                return jsonify({
                    'status': 'error',
                    'message': 'Audio too quiet'
                }), 400
            
            # Extract features
            features = audio_extractor.extract_features(audio_data, sample_rate)
            if features is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Feature extraction failed'
                }), 500
            
            # Reshape features
            features = features.reshape(1, features.shape[0], features.shape[1])
            
            # Get emotion prediction
            prediction = loaded_model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction[0])
            emotion_confidence = float(prediction[0][predicted_class])
            predicted_emotion = str(EMOTION_LABELS[predicted_class])  # Ensure string type
            
            # Get sentiment analysis
            sentiment_result = sentiment_analyzer.analyze(audio_data, sample_rate)
            
            # Normalize emotion and adjust based on sentiment
            predicted_emotion = normalize_emotion(predicted_emotion)
            if sentiment_result.get('emergency_detected', False):
                if predicted_emotion == 'happy':
                    # Look for alternative emotions
                    sorted_predictions = np.argsort(prediction[0])[::-1]
                    alternative_emotions = ['fear', 'angry', 'sad']
                    
                    for idx in sorted_predictions[1:]:
                        current_emotion = normalize_emotion(EMOTION_LABELS[idx])
                        if current_emotion in alternative_emotions:
                            predicted_class = idx
                            predicted_emotion = current_emotion
                            emotion_confidence = float(prediction[0][idx]) * 1.2
                            break
            
            # Get scream detection result
            logger.debug(f"Calling scream detector with emotion: {predicted_emotion} ({type(predicted_emotion)})")
            scream_result = scream_detector.detect_scream(
                predicted_emotion=predicted_emotion,
                emotion_confidence=emotion_confidence,
                sentiment_result=sentiment_result
            )

            # Prepare response
            response_data = {
                'status': 'success',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'processing_time': (datetime.now(timezone.utc) - start_time).total_seconds(),
                'emotion': {
                    'label': predicted_emotion,
                    'confidence': emotion_confidence,
                    'distribution': {
                        str(emotion): float(conf) 
                        for emotion, conf in zip(EMOTION_LABELS, prediction[0])
                    }
                },
                'sentiment': sentiment_result,
                'scream_detection': scream_result,
                'audio_metrics': {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'max_amplitude': float(np.max(np.abs(audio_data)))
                }
            }

            return jsonify(response_data)
            
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Processing failed: {str(e)}',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if all([loaded_model, scream_detector, sentiment_analyzer, audio_extractor]) else 'degraded',
        'components': {
            'emotion_model': loaded_model is not None,
            'sentiment_analyzer': sentiment_analyzer is not None,
            'scream_detector': scream_detector is not None,
            'audio_extractor': audio_extractor is not None
        },
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'user': CURRENT_USER,
        'uptime': TIMESTAMP_UTC
    }
    return jsonify(status)

@app.route('/metrics')
def metrics():
    """Basic metrics endpoint"""
    return jsonify({
        'model_info': {
            'input_shape': MODEL_CONFIG['input_shape'],
            'n_classes': MODEL_CONFIG['n_classes'],
            'emotion_labels': [str(e) for e in EMOTION_LABELS]  # Ensure all labels are strings
        },
        'system_status': {
            'model_loaded': loaded_model is not None,
            'components_ready': all([loaded_model, scream_detector, sentiment_analyzer, audio_extractor])
        },
        'user_info': {
            'current_user': CURRENT_USER,
            'last_startup': TIMESTAMP_UTC
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)