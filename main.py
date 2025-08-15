
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import time
import onnxruntime

# -------------------------------
# Load ONNX model (CPU or GPU)
# -------------------------------
@st.cache_resource
def load_onnx_model(use_gpu=False):
    providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession("fasterrcnn_mask_detector.onnx", providers=providers)
    return session

# -------------------------------
# Preprocess image for ONNX model
# -------------------------------
def preprocess_image_np(image, input_size=(224,224)):
    # Convert PIL Image to NumPy array
    img = np.array(image.convert("RGB"))
    # Resize to model input size using OpenCV
    img_resized = cv2.resize(img, input_size)
    # Normalize to [0,1] and transpose to C,H,W
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    # Add batch dimension
    return np.expand_dims(img_tensor, axis=0)

# -------------------------------
# ONNX prediction
# -------------------------------
def predict_image_onnx(image, session, input_size=(224,224)):
    img_numpy = preprocess_image_np(image, input_size)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_numpy})

    # Handle empty outputs safely
    boxes = outputs[0] if outputs[0].size > 0 else np.empty((0, 4))
    labels = outputs[1] if outputs[1].size > 0 else np.empty((0,))
    scores = outputs[2] if outputs[2].size > 0 else np.empty((0,))

    prediction = {
        'boxes': torch.tensor(np.atleast_2d(boxes)),
        'labels': torch.tensor(np.atleast_1d(labels)),
        'scores': torch.tensor(np.atleast_1d(scores))
    }
    return prediction

# -------------------------------
# Draw predictions on original image
# -------------------------------
def draw_predictions(original_image, prediction, model_input_size=(224,224), threshold=0.5):
    img_array = np.array(original_image.convert("RGB"))
    orig_h, orig_w = img_array.shape[:2]
    input_w, input_h = model_input_size

    boxes = np.atleast_2d(prediction['boxes'].numpy())
    scores = np.atleast_1d(prediction['scores'].numpy())
    labels = np.atleast_1d(prediction['labels'].numpy())

    if boxes.shape[0] == 0:
        return img_array  # no detections

    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            # scale box to original image size
            x1, y1, x2, y2 = box
            x1 = int(x1 / input_w * orig_w)
            y1 = int(y1 / input_h * orig_h)
            x2 = int(x2 / input_w * orig_w)
            y2 = int(y2 / input_h * orig_h)

            if label == 1:
                color = (0, 255, 0)
                label_text = "Mask"
            elif label == 2:
                color = (0, 0, 255)
                label_text = "No Mask"
            elif label == 3:
                color = (255, 165, 0)
                label_text = "Incorrect Mask"
            else:
                continue

            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_array, f"{label_text} {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_array

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.title("Face Mask Detection")
    st.write("Upload an image or use your webcam to detect face masks!")

    use_gpu = st.sidebar.checkbox("Use GPU (CUDA) if available")
    session = load_onnx_model(use_gpu)

    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Upload Image", "Webcam"])

    if app_mode == "About":
        st.markdown("This application uses a Faster R-CNN ONNX model to detect face masks.")
        st.markdown("**How to use:**")
        st.markdown("1. Select app mode (Upload Image/Webcam)")
        st.markdown("2. Upload an image or allow webcam access")
        st.markdown("3. The model will detect masks in real-time or on the uploaded image")

    elif app_mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Masks"):
                start = time.time()
                prediction = predict_image_onnx(image, session)
                result_image = draw_predictions(image, prediction)
                st.image(result_image, caption="Detection Result", use_column_width=True)
                st.write(f"Prediction time: {time.time() - start:.2f}s")

    elif app_mode == "Webcam":
        st.write("Ensure you’ve given camera access to the browser!")
        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            ret, frame = camera.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            start_time = time.time()
            prediction = predict_image_onnx(image, session)
            result_image = draw_predictions(image, prediction)
            FRAME_WINDOW.image(result_image)
            st.write(f"Inference time: {time.time() - start_time:.2f}s")

        camera.release()

if __name__ == "__main__":
    main()