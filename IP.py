import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Processing Demo", layout="wide")
st.title("🖼️ Image Processing Playground")

# --- Utility functions ---
def show_histogram(img, title="Histogram"):
    fig, ax = plt.subplots()
    if len(img.shape) == 2:  # grayscale
        ax.hist(img.ravel(), bins=256, range=(0, 256), color="black")
    else:  # color
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            ax.hist(img[:, :, i].ravel(), bins=256, range=(0, 256), color=col, alpha=0.5)
    ax.set_title(title)
    st.pyplot(fig)

# --- Transformations ---
def negative(img):
    return 255 - img

def contrast_stretching(img, r1, s1, r2, s2):
    def stretch(pixel):
        if pixel < r1:
            return (s1 / r1) * pixel
        elif pixel < r2:
            return ((s2 - s1) / (r2 - r1)) * (pixel - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pixel - r2) + s2
    return np.vectorize(stretch)(img).astype(np.uint8)

def piecewise_linear(img, points):
    xp, fp = zip(*points)
    return np.interp(img.flatten(), xp, fp).reshape(img.shape).astype(np.uint8)

def log_transform(img, c):
    img_norm = img / 255.0
    log_img = c * np.log1p(img_norm)
    log_img = np.uint8(255 * log_img / np.max(log_img))
    return log_img

def gamma_transform(img, gamma):
    img_norm = img / 255.0
    gamma_img = np.power(img_norm, gamma)
    return np.uint8(255 * gamma_img)

def hist_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def adaptive_hist_equalization(img, clip_limit=2.0, tile_size=8):
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(img)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Image Type", ["Grayscale", "Color"])
method = st.sidebar.selectbox(
    "Select Method",
    [
        "Negative",
        "Contrast Stretching",
        "Piecewise Linear",
        "Log Transformation",
        "Gamma Transformation",
        "Histogram Equalization",
        "Adaptive Histogram Equalization / CLAHE"
    ]
)

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if mode == "Grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    processed = None

    if method == "Negative":
        processed = negative(img)

    elif method == "Contrast Stretching":
        r1 = st.sidebar.slider("r1", 0, 255, 50)
        s1 = st.sidebar.slider("s1", 0, 255, 0)
        r2 = st.sidebar.slider("r2", 0, 255, 200)
        s2 = st.sidebar.slider("s2", 0, 255, 255)
        processed = contrast_stretching(img, r1, s1, r2, s2)

    elif method == "Piecewise Linear":
        st.sidebar.write("Define piecewise points (x,y):")
        points = [(0, 0), (128, 100), (255, 255)]
        processed = piecewise_linear(img, points)

    elif method == "Log Transformation":
        c = st.sidebar.slider("c (scale)", 1, 10, 5)
        processed = log_transform(img, c)

    elif method == "Gamma Transformation":
        gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0, 0.1)
        processed = gamma_transform(img, gamma)

    elif method == "Histogram Equalization":
        processed = hist_equalization(img)

    elif method == "Adaptive Histogram Equalization / CLAHE":
        clip = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0)
        tile = st.sidebar.slider("Tile Size", 2, 16, 8)
        processed = adaptive_hist_equalization(img, clip, tile)

    # --- Display ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        st.image(img, channels="BGR" if mode == "Color" else "GRAY")
        show_histogram(img, "Histogram Before")

    with col2:
        st.subheader("After")
        st.image(processed, channels="BGR" if mode == "Color" else "GRAY")
        show_histogram(processed, "Histogram After")
