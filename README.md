# Traffic Sign Recognition using VGG16

Deep learning project for **image classification of 58 traffic sign classes**, with an interactive **Streamlit** interface.

## Overview

This project uses **transfer learning / fine-tuning of VGG16** (ImageNet pretrained) with TensorFlow/Keras. Users can upload a traffic sign image to the Streamlit app and obtain the predicted class and model score.

## Tech stack

- Python, TensorFlow / Keras, VGG16
- OpenCV, NumPy, Pillow
- Streamlit
- Jupyter Notebook

## Approach

1. Dataset: approximately **4,047 images** (as described in the project presentation).
2. Data split: **80% training, 10% validation, 10% testing**.
3. Preprocessing: resize to **224 × 224**, normalize RGB pixel values to [0, 1].
4. Training augmentation: shifts, brightness changes, zoom, shear and rotation.
5. Classification: fine-tuned VGG16 backbone followed by a custom fully connected classifier for **58 classes**.
6. Inference: Streamlit app shows the predicted class and prediction score.

## Run the app locally

> Compatibility note: the model was saved with **Keras 2.10**. For best compatibility, use **Python 3.10**, TensorFlow 2.10.1 and the versions from `requirements.txt`. The saved model has not been runtime-tested in this packaged repository.

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
streamlit run app.py
```

**Windows activation:** `.venv\Scripts\activate`

The application loads `model.json` (architecture) and `model.h5` (weights). Both must remain in the repository root.

## Repository structure

```text
traffic-sign-recognition/
├── app.py                        # Streamlit inference interface
├── train_model.py                # Original training script
├── model.json                    # Saved model architecture
├── model.h5                      # Saved trained weights (~77 MB)
├── notebooks/
│   ├── traffic_sign_recognition.ipynb
│   └── report_original.ipynb
├── docs/
│   └── SPEECH.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## Training and reproduction

The training script and notebooks contain **original local Windows paths** to the dataset, which **is not included**. Update those paths to your own dataset before retraining. The app can be used directly with the included trained model (subject to environment compatibility).

## Limitations

- This is a traffic sign **image classification** project, not a full object detection or real-time traffic scene recognition system.
- No validated accuracy figure is included here: refer to the notebook results for experiment-specific metrics.
- The class names shown by the Streamlit app come from its built-in label mapping; verify them against the actual dataset label order before relying on predictions.
- The optional university logo referenced in the original code was not provided, so the packaged app skips it when absent.

## Contributors

Amine Chelabi, Youcef Brahimi, Mohammed Bouziane Belbekri (as credited in the supplied training code).
