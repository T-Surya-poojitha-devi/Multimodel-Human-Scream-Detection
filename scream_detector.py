import logging
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ScreamDetector:
    def __init__(self):
        # Current user and timestamp
        self.current_user = "FaheemKhan0817"
        self.timestamp = "2025-01-30 05:35:18"
        
        self.emergency_phrases = {
            'high_priority': [
                'help', 'emergency', 'stop', 'no', 'please', 
                'hurt', 'kill', 'die', 'pain', 'fire',
                'police', 'danger', 'blood', 'scared', 'fear',
                'scream', 'crying', 'shouting', 'yelling'
            ],
            'medium_priority': [
                'loud', 'noise', 'afraid', 'worried',
                'frightened', 'terrified', 'panic', 'help me',
                'scared', 'crying', 'screaming', 'shouting'
            ]
        }
        
        # Adjusted weights for better sensitivity
        self.emotion_weights = {
            'fear': 0.85,     # Increased sensitivity
            'angry': 0.80,    # Increased sensitivity
            'sad': 0.70,      # Increased for crying detection
            'surprised': 0.60,
            'happy': -0.3,    # Negative weight for happy emotions
            'neutral': 0.0,
            'unknown': 0.4    # Default case with moderate weight
        }

        # Volume thresholds for detection
        self.volume_thresholds = {
            'scream': 0.75,
            'shout': 0.60,
            'normal': 0.40
        }

    def normalize_emotion(self, emotion):
        """Ensure emotion is a valid string in lowercase"""
        try:
            if emotion is None:
                return 'unknown'
            return str(emotion).lower().strip()
        except:
            return 'unknown'

    def detect_scream(self, predicted_emotion, emotion_confidence, sentiment_result):
        """Enhanced detection for screams, shouts, and crying"""
        try:
            logger.info(f"Starting scream detection for user {self.current_user}")
            
            # Normalize inputs
            predicted_emotion = self.normalize_emotion(predicted_emotion)
            emotion_confidence = float(emotion_confidence)
            
            # Initialize detection variables
            emergency_detected = False
            detected_phrases = []
            high_priority_detected = False
            
            # Check speech content
            if sentiment_result and sentiment_result.get('status') == 'success':
                text = str(sentiment_result.get('text', '')).lower()
                
                # Detect emergency phrases
                high_priority_phrases = [
                    phrase for phrase in self.emergency_phrases['high_priority'] 
                    if phrase in text
                ]
                medium_priority_phrases = [
                    phrase for phrase in self.emergency_phrases['medium_priority'] 
                    if phrase in text
                ]
                
                detected_phrases = high_priority_phrases + medium_priority_phrases
                emergency_detected = len(detected_phrases) > 0
                high_priority_detected = len(high_priority_phrases) > 0

            # Calculate base emotion score
            emotion_weight = self.emotion_weights.get(predicted_emotion, self.emotion_weights['unknown'])
            base_emotion_score = emotion_weight * min(max(emotion_confidence, 0.0), 1.0)

            # Boost score for high confidence negative emotions
            if emotion_confidence > 0.7 and emotion_weight > 0.5:
                base_emotion_score *= 1.3

            # Calculate sentiment contribution
            sentiment_score = 0.0
            if sentiment_result and sentiment_result.get('status') == 'success':
                polarity = float(sentiment_result.get('polarity', 0))
                sentiment_score = -polarity * 0.7  # Negative polarity increases score
                
                # Boost score for emergency keywords
                if emergency_detected:
                    sentiment_score = max(sentiment_score * 1.4, 0.75)

            # Calculate final distress score
            distress_score = max(
                base_emotion_score,
                sentiment_score
            )

            # Apply emergency modifiers
            if emergency_detected:
                distress_score = max(distress_score + 0.35, 0.75)
            if high_priority_detected:
                distress_score = max(distress_score + 0.45, 0.85)

            # Get detection type
            detection_type = self.get_detection_type(
                predicted_emotion, 
                emotion_confidence,
                distress_score,
                high_priority_detected
            )

            response = {
                'is_scream': distress_score > 0.5 or emergency_detected,
                'detection_type': detection_type,
                'confidence': min(distress_score * 1.2, 1.0),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'user': self.current_user,
                'details': {
                    'emotion': {
                        'label': predicted_emotion,
                        'confidence': float(emotion_confidence),
                        'weight': float(emotion_weight)
                    },
                    'emergency_phrases_detected': detected_phrases,
                    'distress_score': float(distress_score),
                    'base_emotion_score': float(base_emotion_score),
                    'sentiment_score': float(sentiment_score),
                    'is_high_priority': bool(high_priority_detected)
                }
            }

            logger.info(f"Detection completed for {self.current_user}: {detection_type}")
            return response

        except Exception as e:
            logger.error(f"Error in distress detection: {str(e)}", exc_info=True)
            return {
                'is_scream': False,
                'detection_type': None,
                'confidence': 0,
                'reason': 'error',
                'details': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'user': self.current_user
            }

    def get_detection_type(self, emotion, confidence, distress_score, high_priority):
        """Determine type of distress with improved sensitivity"""
        try:
            emotion = self.normalize_emotion(emotion)
            confidence = float(confidence)
            distress_score = float(distress_score)

            # More sensitive thresholds
            if high_priority or distress_score > 0.70:
                return 'scream'
            elif emotion in ['angry', 'fear'] and confidence > 0.55:
                return 'shouting'
            elif emotion == 'sad' and confidence > 0.45:
                return 'crying'
            elif distress_score > 0.45:
                return 'distress'
            return 'normal'
            
        except Exception as e:
            logger.error(f"Error in get_detection_type: {str(e)}")
            return 'normal'

    def analyze_volume(self, audio_features):
        """Analyze volume patterns for scream detection"""
        try:
            rms = np.mean(np.abs(audio_features), axis=0)
            peak_volume = np.max(rms)
            avg_volume = np.mean(rms)
            
            # Calculate volume score
            volume_score = 0.0
            
            if peak_volume > self.volume_thresholds['scream']:
                volume_score = 0.9
            elif peak_volume > self.volume_thresholds['shout']:
                volume_score = 0.7
            elif peak_volume > self.volume_thresholds['normal']:
                volume_score = 0.5
                
            return volume_score
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return 0.0