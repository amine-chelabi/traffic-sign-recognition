from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import model_from_json

st.set_page_config(page_title="Traffic Sign Recognition", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent

# Labels copied from the original application. Their correspondence to
# training classes has not yet been independently verified.
LABELS = {0: 'Speed limit (5km/h)', 1: 'Speed limit (15km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (40km/h)', 4: 'Speed limit (50km/h)', 5: 'Speed limit (60km/h)', 6: 'Speed limit (70km/h)', 7: 'speed limit (80km/h)', 8: 'Dont Go straight or left', 9: 'Dont Go straight or Right', 10: 'Dont Go straight', 11: 'Dont Go Left', 12: 'Dont Go Left or Right', 13: 'Dont Go Right', 14: 'Dont overtake from Left', 15: 'No Uturn', 16: 'No Car', 17: 'No horn', 18: 'Speed limit (40km/h)', 19: 'Speed limit (50km/h)', 20: 'Go straight or right', 21: 'Go straight', 22: 'Go Left', 23: 'Go Left or right', 24: 'Go Right', 25: 'keep Left', 26: 'keep Right', 27: 'Roundabout mandatory', 28: 'watch out for cars', 29: 'Horn', 30: 'Bicycles crossing', 31: 'Uturn', 32: 'Road Divider', 33: 'Traffic signals', 34: 'Danger Ahead', 35: 'Zebra Crossing', 36: 'Bicycles crossing', 37: 'Children crossing', 38: 'Dangerous curve to the left', 39: 'Dangerous curve to the right', 40: 'Unknown1', 41: 'Unknown2', 42: 'Unknown3', 43: 'Go right or straight', 44: 'Go left or straight', 45: 'Unknown4', 46: 'ZigZag Curve', 47: 'Train Crossing', 48: 'Under Construction', 49: 'Unknown5', 50: 'Fences', 51: 'Heavy Vehicle Accidents', 52: 'Unknown6', 53: 'Give Way', 54: 'No stopping', 55: 'No entry', 56: 'Unknown7', 57: 'Unknown8'}


@st.cache_resource
def load_model():
    architecture_path = PROJECT_DIR / "model.json"
    weights_path = PROJECT_DIR / "model.h5"
    with architecture_path.open("r", encoding="utf-8") as f:
        model = model_from_json(f.read())
    model.load_weights(str(weights_path))
    return model


def predict_top5(image, model):
    # Same preprocessing as the original application: RGB, center crop to
    # 224 x 224 and scale pixel values from 0–255 to 0–1.
    resized = ImageOps.fit(image.convert("RGB"), (224, 224), Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    scores = model.predict(array[np.newaxis, ...], verbose=0)[0]
    top_indices = np.argsort(scores)[::-1][:5]
    return [(int(i), float(scores[i])) for i in top_indices]


st.title("Traffic Sign Recognition")
st.write("Traffic sign classification with a VGG16-based neural network.")

try:
    with st.spinner("Loading VGG16 model..."):
        model = load_model()
except Exception as exc:
    st.error("Unable to load the model. Check model.json and model.h5.")
    st.exception(exc)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a traffic sign image", type=["jpg", "jpeg", "png"]
)
if uploaded_file is None:
    st.write("Choose an image to display the five highest-scoring classes.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        results = predict_top5(image, model)
    except Exception as exc:
        st.error("Unable to process this image.")
        st.exception(exc)
        st.stop()

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_column_width=True)
    with right:
        top_class, top_score = results[0]
        st.subheader("Top prediction")
        st.write("Class index:", top_class)
        st.write("Unverified label:", LABELS.get(top_class, "Unknown"))
        st.write("Model score:", f"{top_score * 100:.2f}%")
        st.subheader("Top 5 predictions")
        for rank, (class_index, score) in enumerate(results, start=1):
            st.write(
                f"{rank}. Class {class_index} — "
                f"{LABELS.get(class_index, 'Unknown')} — {score * 100:.2f}%"
            )
            st.progress(min(max(score, 0.0), 1.0))
        st.caption(
            "Scores are softmax outputs, not measured accuracy. "
            "Labels need verification against the training dataset."
        )
