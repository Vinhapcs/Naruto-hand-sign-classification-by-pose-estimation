# Naruto Hand Sign Classification

A machine learning project that classifies Naruto hand signs using pose estimation and Random Forest classification.

## Project Overview

This project identifies and classifies the 12 basic hand signs from the Naruto universe using computer vision and machine learning techniques. By leveraging pose estimation for feature extraction and Random Forest for classification, the system can accurately recognize hand signs from images captured at various angles.

## Dataset

- **Labels**: 12 different Naruto hand signs
- **Size**: Less than 400 images per class
- **Variety**: Images captured from different angles to ensure model robustness

## Technical Approach

### Pose Estimation for Feature Extraction

Instead of using raw image data, this project employs pose estimation to extract key points from hands, which offers several advantages:

- **Works well with smaller datasets**: Reduces the need for massive amounts of training data
- **Translation and rotation invariance**: Features remain consistent regardless of hand position or angle
- **Dimensionality reduction**: Simplifies the feature space by focusing only on relevant joint positions
- **Faster training**: Significantly reduces computational requirements compared to deep learning approaches

### Random Forest Classification

After extracting pose features, a Random Forest classifier is used to identify the hand signs:

- **Simple yet powerful**: Achieves high accuracy without complex hyperparameter tuning
- **Fast training**: Can be trained quickly on modest hardware
- **Robust to overfitting**: Works well with the limited dataset size
- **Interpretable**: Feature importance can be analyzed to understand decision-making

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/naruto-hand-sign-classification.git
cd naruto-hand-sign-classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data_path /path/to/dataset --save_model model.pkl
```

### Prediction

```bash
python predict.py --model model.pkl --image /path/to/image.jpg
```

### Real-time Classification

```bash
python realtime.py --model model.pkl
```

## Project Structure

```
├── data/
│   ├── raw/               # Raw image dataset
│   └── processed/         # Processed pose estimation features
├── models/                # Saved model files
├── notebooks/            
│   └── exploration.ipynb  # Data exploration and model testing
├── src/
│   ├── data_processing.py # Dataset preparation scripts
│   ├── pose_extraction.py # Pose estimation feature extraction
│   ├── train.py           # Model training script
│   ├── predict.py         # Single image prediction
│   └── realtime.py        # Real-time webcam classification
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Performance

- **Accuracy**: ~92% on test set
- **Training time**: Less than 5 minutes on standard hardware
- **Inference speed**: Suitable for real-time applications

## Future Improvements

- Expand the dataset with more variations in lighting and backgrounds
- Experiment with ensemble methods to further improve accuracy
- Add a confidence score for predictions
- Implement sequence recognition for multi-sign jutsus

## License

[MIT License](LICENSE)

## Acknowledgements

- The [MediaPipe](https://google.github.io/mediapipe/) library for pose estimation
- Scikit-learn for the Random Forest implementation
- The Naruto franchise for inspiration