import librosa
import numpy as np
import logging
import scipy.signal as signal
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate
        self.max_pad_len = 862    # Maximum padding length
        self.n_mfcc = 40          # Number of MFCC features
        
        # Enhanced parameters for better detection
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.fmin = 20
        self.fmax = 8000

    def preprocess_audio(self, audio_data, sample_rate):
        """Preprocess audio data with noise reduction and normalization"""
        try:
            # Resample if necessary
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)

            # Apply pre-emphasis filter
            audio_data = librosa.effects.preemphasis(audio_data)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)

            # Apply noise reduction
            noise_reduced = self.reduce_noise(audio_data)
            
            return noise_reduced
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {str(e)}")
            return None

    def reduce_noise(self, audio_data):
        """Apply noise reduction to the audio signal"""
        try:
            # Estimate noise from the first 1000 samples
            noise_sample = audio_data[:1000]
            noise_power = np.mean(noise_sample ** 2)
            
            # Apply simple noise gate
            threshold = noise_power * 2
            audio_data = np.where(audio_data**2 > threshold, audio_data, 0)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {str(e)}")
            return audio_data

    def extract_features(self, audio_data, sample_rate):
        """Extract audio features with enhanced sensitivity"""
        try:
            logger.info(f"Extracting features from audio: length={len(audio_data)}, sr={sample_rate}")
            
            if len(audio_data) == 0:
                logger.error("Empty audio data")
                return None

            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data, sample_rate)
            if processed_audio is None:
                return None

            # Calculate volume features
            rms = librosa.feature.rms(y=processed_audio)[0]
            peak_volume = np.max(rms)
            avg_volume = np.mean(rms)
            
            logger.info(f"Audio volume metrics - Peak: {peak_volume:.4f}, Average: {avg_volume:.4f}")

            # Extract MFCCs with enhanced parameters
            mfccs = librosa.feature.mfcc(
                y=processed_audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )

            # Extract additional features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=processed_audio, 
                sr=self.sample_rate
            )
            
            # Enhance features based on volume
            volume_scale = self.calculate_volume_scaling(peak_volume, avg_volume)
            mfccs *= volume_scale

            # Handle padding
            if mfccs.shape[1] > self.max_pad_len:
                mfccs = mfccs[:, :self.max_pad_len]
            else:
                pad_width = self.max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

            logger.info(f"Extracted features shape: {mfccs.shape}")
            return mfccs

        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            return None

    def calculate_volume_scaling(self, peak_volume, avg_volume):
        """Calculate scaling factor based on volume metrics"""
        if peak_volume > 0.8:  # Very loud sounds
            return 1.4
        elif peak_volume > 0.6:  # Loud sounds
            return 1.2
        elif peak_volume > 0.4:  # Moderate sounds
            return 1.0
        else:  # Quiet sounds
            return 0.8

def extract_features(audio_data, sample_rate, max_pad_len=862):
    """Wrapper function for compatibility"""
    extractor = AudioFeatureExtractor()
    return extractor.extract_features(audio_data, sample_rate)