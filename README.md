# Speaker-Identification-System


This project is an end-to-end **speaker identification system** that determines whether an uploaded audio file contains the voice of a specific target speaker (**Rowdy Delaney**) or not.

The system combines:
- Audio signal processing (Mel-spectrograms)
- A convolutional neural network (CNN) trained for binary speaker classification
- A Streamlit-based web interface for easy interaction

---

## 📌 Project Overview

The pipeline works as follows:

1. User uploads an audio file (`.wav`, `.mp3`, or `.flac`)
2. Audio is converted into a Mel-spectrogram image
3. The spectrogram is normalized and resized
4. A trained CNN model performs inference
5. The system outputs:
   - Predicted speaker class (Target / Non-Target)
   - Model confidence score

This approach treats speaker identification as an **image-based binary classification problem**.

---

## 🧠 Model Training Details

- **Dataset**:  
  The model was trained using **Mel-spectrogram images derived from the LibriSpeech audio dataset**.

- **Preprocessing Pipeline**:
  - Raw LibriSpeech audio files were split into train, validation, and test sets **before feature extraction** to prevent data leakage.
  - Each audio sample was converted into a Mel-spectrogram using `librosa`.
  - Spectrograms were saved as RGB images and resized to **128 × 128**.
  - These images served as direct input to the CNN.

- **Architecture**:
  - 3-block CNN (Conv → BatchNorm → ReLU → MaxPool)
  - Binary output neuron

- **Loss Function**:  
  `BCEWithLogitsLoss`

- **Classification Task**:  
  Binary speaker identification (Target speaker vs. all others)



