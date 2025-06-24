# 🎤 Multimodal Human Scream Detection System

This project presents a real-time **Human Scream Detection System** that uses both **audio and textual cues** to accurately detect distress situations. The system integrates **audio signal processing**, **speech-to-text conversion**, **sentiment/emotion analysis**, and a **fusion-based classification model** to identify emergency screams effectively in noisy environments.

## 🚀 Features

- 🎙️ Real-time audio recording from microphone
- 🔊 MFCC & Spectrogram-based audio feature extraction
- 🤖 Audio-based scream classification using CNN/LSTM
- 💬 Text extraction using SpeechRecognition + Google Speech-to-Text API
- 🧠 Sentiment & emotion classification using BERT model
- 🔀 Fusion model to combine audio + text decisions
- ✅ Distress alert triggered for emergency conditions
- 🌐 Flask-based web interface with live predictions

## 🧰 Tech Stack

- Python (Flask, Librosa, NumPy, Scikit-learn, TensorFlow/Keras)
- SpeechRecognition + Google STT API
- BERT (Hugging Face Transformers)
- HTML/CSS + JavaScript (Frontend)
- Jupyter Notebook (Model Training & Testing)

## 📁 Directory Structure

multimodal_scream_detection/
├── app.py
├── audio_processor.py
├── scream_detector.py
├── requirements.txt
├── templates/
│ └── index.html
├── static/
│ └── js/
│ └── app.js
├── converted_model/
│ ├── emotion_model.h5
│ └── scream_detection_model.h5
└── README.md

- `app.py`: The main Flask application file that handles HTTP requests and initializes the backend components.
- `audio_processor.py`: Contains functions for audio feature extraction and preprocessing.
- `scream_detector.py`: Contains the ScreamDetector class for detecting distress signals in audio.
- `requirements.txt`: Lists all the dependencies required for the project.
- `templates/index.html`: The HTML template for the frontend interface.
- `static/js/app.js`: The JavaScript file for handling frontend interactions and audio recording.
- `converted_model/`: Directory containing the pretrained models.
- `emotion_model.h5`: The pretrained model for emotion classification.
- `scream_detection_model.h5`: The pretrained model for scream detection.
- `README.md`: Documentation file for the project.

## execution video sample 

https://github.com/user-attachments/assets/7f213aea-82ea-4ecf-9516-c75f9efa1a5f

## Dependencies

The project requires the following Python libraries:

- Flask==2.2.3
- flask-cors==3.0.10
- librosa==0.10.2.post1
- numpy==1.24.3
- tensorflow==2.13.0
- scipy==1.13.1
- numba==0.60.0
- requests==2.32.3

## 👥 Acknowledgments

- [Librosa](https://librosa.org/) for audio processing  
- [Hugging Face Transformers](https://huggingface.co/transformers/) for BERT  
- [TensorFlow/Keras](https://www.tensorflow.org/) for modeling  
- [Google Speech-to-Text API](https://cloud.google.com/speech-to-text)

  ## 📌 Use Cases

- Smart surveillance and public safety systems  
- Emergency detection in smart cities  
- Healthcare monitoring for elderly or patients  
- Voice-based security systems  

  ## Installation and Setup

### Step 1: Download and Install Miniconda

Miniconda is a free minimal installer for conda. It is used to manage environments and packages.

1. Download Miniconda for your operating system from the official website: [Miniconda Download Page](https://docs.conda.io/en/latest/miniconda.html)
2. Install Miniconda by following the instructions provided on the download page.

### Step 2: Create and Activate a Conda Environment

Open a terminal (or Anaconda Prompt on Windows) and create a new conda environment with Python 3.9.21:

```sh
conda create --name multimodal_scream_detection python=3.9.21
conda activate multimodal_scream_detection

Step 3: Install Project Dependencies
Navigate to the project directory (e.g., C:\test\multimodal_scream_detection) and install the dependencies listed in requirements.txt:

```sh

cd C:\test\multimodal_scream_detection
pip install -r requirements.txt

Start the Flask App -** python app.py**






