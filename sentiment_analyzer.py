# sentiment_analyzer.py

import speech_recognition as sr
from textblob import TextBlob
import numpy as np
import wave
import tempfile
import os
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.emergency_keywords = {
            # Crying-related keywords
            'cry': -0.7, 'crying': -0.7, 'sobbing': -0.8, 'weeping': -0.7,
            'tears': -0.6, 'bawling': -0.8,
            
            # Shouting-related keywords
            'shout': -0.6, 'yell': -0.6, 'screaming': -0.8, 'yelling': -0.7,
            'shouting': -0.7, 'loud': -0.5,
            
            # Emergency keywords
            'help': -0.7, 'emergency': -0.8, 'danger': -0.8,
            'hurt': -0.7, 'die': -0.9, 'death': -0.9, 'attack': -0.8,
            'scared': -0.6, 'threat': -0.7, 'panic': -0.7, 'violence': -0.8,
            'weapon': -0.8, 'blood': -0.7, 'injury': -0.7, 'crime': -0.8,
            'assault': -0.9, 'harm': -0.7, 'threatening': -0.7, 'unsafe': -0.6,
            'kill': -0.9, 'murder': -0.9, 'police': -0.7, 'fire': -0.8,
            'shot': -0.8, 'shooting': -0.8, 'gun': -0.8, 'knife': -0.8,
            'pain': -0.7, 'stop': -0.6, 'no': -0.5, 'please': -0.4
        }
        
        # Initialize speech recognizer with optimized settings
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    @contextmanager
    def temp_wav_file(self):
        """Context manager for temporary WAV file handling"""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            yield temp_path
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {e}")

    def analyze(self, audio_data, sample_rate):
        """Analyze sentiment from audio data"""
        try:
            with self.temp_wav_file() as temp_path:
                # Save audio data
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                # Perform speech recognition
                with sr.AudioFile(temp_path) as source:
                    audio = self.recognizer.record(source)
                    try:
                        text = self.recognizer.recognize_google(audio, language='en-US')
                    except sr.UnknownValueError:
                        return {
                            'status': 'warning',
                            'message': 'Could not understand speech',
                            'text': 'Speech not recognized',
                            'polarity': 0.0,
                            'subjectivity': 0.0,
                            'label': 'Neutral',
                            'emergency_keywords_detected': []
                        }
                    except sr.RequestError as e:
                        logger.error(f"Speech recognition service error: {str(e)}")
                        return {
                            'status': 'error',
                            'message': f'Speech recognition service error: {str(e)}',
                            'text': 'Service error',
                            'polarity': 0.0,
                            'subjectivity': 0.0,
                            'label': 'Error',
                            'emergency_keywords_detected': []
                        }

                    # Perform sentiment analysis
                    analysis = TextBlob(text)
                    base_polarity = float(analysis.sentiment.polarity)
                    
                    # Check for emergency keywords
                    words = text.lower().split()
                    emergency_words = [word for word in words if word in self.emergency_keywords]
                    
                    # Calculate adjusted polarity
                    if emergency_words:
                        keyword_scores = [self.emergency_keywords[word] for word in emergency_words]
                        emergency_score = sum(keyword_scores) / len(keyword_scores)
                        # Weight emergency keywords more heavily
                        adjusted_polarity = (base_polarity + (emergency_score * 2)) / 3
                    else:
                        adjusted_polarity = base_polarity

                    # Get final sentiment label
                    sentiment_label = self.get_sentiment_label(adjusted_polarity)
                    
                    # Override sentiment if emergency words are detected
                    if emergency_words and sentiment_label in ['Neutral', 'Positive', 'Very Positive']:
                        sentiment_label = 'Negative'
                        adjusted_polarity = -0.6

                    return {
                        'status': 'success',
                        'text': text,
                        'polarity': adjusted_polarity,
                        'original_polarity': base_polarity,
                        'subjectivity': float(analysis.sentiment.subjectivity),
                        'label': sentiment_label,
                        'emergency_keywords_detected': emergency_words,
                        'emergency_detected': bool(emergency_words)
                    }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Analysis error: {str(e)}',
                'text': 'Analysis error',
                'polarity': 0.0,
                'subjectivity': 0.0,
                'label': 'Error',
                'emergency_keywords_detected': []
            }

    def get_sentiment_label(self, polarity):
        """Convert polarity score to sentiment label"""
        if polarity > 0.75:
            return 'Very Positive'
        elif polarity > 0.25:
            return 'Positive'
        elif polarity > -0.25:
            return 'Neutral'
        elif polarity > -0.75:
            return 'Negative'
        else:
            return 'Very Negative'