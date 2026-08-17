"""
app.py
------
GestureSense: Touchless Human-Computer Interaction using ANN & MediaPipe
STEP 13 : Application Development (Streamlit)

Pages:
  - Home
  - Project Description
  - Prediction (Upload Image / Webcam)
  - Control Center (gestures trigger real actions: slide nav + media control)

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime

from predictor import GesturePredictor

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GestureSense | Touchless HCI System",
    page_icon="🖐️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-top: 0;
    }
    .gesture-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-box {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .warn-box {
        background-color: #fef2f2;
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return GesturePredictor()


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
st.sidebar.title("🖐️ GestureSense")
st.sidebar.caption("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Project Description", "Prediction", "Control Center"])

GESTURE_EMOJIS = {
    "thumbs_up": "👍", "thumbs_down": "👎", "open_palm": "✋", "fist": "✊",
    "peace": "✌️", "ok_sign": "👌", "pointing": "☝️", "call_me": "🤙",
}

# Maps each gesture to a real, visible action in the Control Center page.
GESTURE_ACTIONS = {
    "thumbs_up": "Next Slide",
    "thumbs_down": "Previous Slide",
    "open_palm": "Pause Media",
    "fist": "Play Media",
    "peace": "Volume Up",
    "ok_sign": "Confirm / Select",
    "pointing": "Mute Toggle",
    "call_me": "Volume Down",
}

# ---------------------------------------------------------------------------
# SESSION STATE (persists across reruns within a browser session)
# ---------------------------------------------------------------------------
if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "volume" not in st.session_state:
    st.session_state.volume = 50
if "is_muted" not in st.session_state:
    st.session_state.is_muted = False
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

SLIDES = [
    ("Slide 1", "Welcome to GestureSense", "#3b82f6"),
    ("Slide 2", "Problem Statement & Business Objective", "#8b5cf6"),
    ("Slide 3", "Dataset & EDA", "#06b6d4"),
    ("Slide 4", "Model Architecture (ANN)", "#10b981"),
    ("Slide 5", "Results & Evaluation", "#f59e0b"),
    ("Slide 6", "Live Demo & Deployment", "#ef4444"),
]


def log_prediction(gesture, confidence, action=None):
    """Append a prediction to the session history log (used for both pages)."""
    st.session_state.prediction_history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Gesture": gesture.replace("_", " ").title(),
        "Confidence": f"{confidence*100:.1f}%",
        "Action Triggered": action or "-",
    })
    st.session_state.prediction_history = st.session_state.prediction_history[:15]


def apply_action(gesture):
    """Applies the gesture's mapped action to the mock slide deck / media player state."""
    action = GESTURE_ACTIONS.get(gesture)
    if action == "Next Slide":
        st.session_state.slide_index = min(st.session_state.slide_index + 1, len(SLIDES) - 1)
    elif action == "Previous Slide":
        st.session_state.slide_index = max(st.session_state.slide_index - 1, 0)
    elif action == "Pause Media":
        st.session_state.is_playing = False
    elif action == "Play Media":
        st.session_state.is_playing = True
    elif action == "Volume Up":
        st.session_state.volume = min(st.session_state.volume + 10, 100)
    elif action == "Volume Down":
        st.session_state.volume = max(st.session_state.volume - 10, 0)
    elif action == "Mute Toggle":
        st.session_state.is_muted = not st.session_state.is_muted
    return action

# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
if page == "Home":
    st.markdown('<p class="main-header">🖐️ GestureSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Touchless Human-Computer Interaction using ANN & MediaPipe</p>', unsafe_allow_html=True)
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", "99.4%")
    col2.metric("Gesture Classes", "8")
    col3.metric("Features Used", "63 landmarks")
    col4.metric("Actions Enabled", "8")

    st.divider()
    st.subheader("Supported Gestures")
    cols = st.columns(4)
    for i, (gesture, emoji) in enumerate(GESTURE_EMOJIS.items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="gesture-card">
                <div style="font-size: 2.5rem;">{emoji}</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">{gesture.replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.info("👉 Head to **Prediction** to test gesture detection with your webcam or an uploaded photo.")
    with col2:
        st.success("🎮 Head to **Control Center** to see gestures trigger real actions — slide navigation & media control.")


# ---------------------------------------------------------------------------
# PROJECT DESCRIPTION PAGE
# ---------------------------------------------------------------------------
elif page == "Project Description":
    st.markdown('<p class="main-header">Project Description</p>', unsafe_allow_html=True)
    st.divider()

    st.subheader("📌 Problem Statement")
    st.write(
        "Traditional device interaction relies on physical touch (keyboards, mice, "
        "touchscreens), which isn't always practical — for accessibility needs, "
        "hygiene-sensitive environments, or hands-free control scenarios. This project "
        "builds a system that recognizes hand gestures in real time from a webcam feed "
        "and maps them to commands, enabling touchless human-computer interaction."
    )

    st.subheader("🎯 Business Objective")
    st.write(
        "Build a lightweight, real-time gesture classifier that can be embedded into "
        "applications for touchless media control, presentation navigation, smart-home "
        "commands, or as an accessibility aid — without requiring specialized hardware, "
        "just a standard webcam."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Input Features")
        st.write(
            "21 hand landmarks detected by MediaPipe Hands, each with (x, y, z) "
            "coordinates → **63 numeric features** per frame."
        )
        st.subheader("📤 Expected Output")
        st.write("Predicted gesture label out of 8 classes, with a confidence score.")

    with col2:
        st.subheader("🎯 Target Variable")
        st.write("`label` — the gesture category (categorical, 8 classes).")
        st.subheader("🛠️ Technologies Used")
        st.write("Python, MediaPipe, TensorFlow/Keras, Optuna, scikit-learn, Streamlit")

    st.divider()
    st.subheader("🔄 Pipeline Workflow")
    st.markdown("""
    1. **Dataset Collection** — Custom webcam-collected hand landmark data (8 gesture classes)
    2. **EDA** — Class balance, feature distributions, correlation, gesture skeleton visualization
    3. **Preprocessing** — Label encoding, train/test split (80:20, stratified), feature scaling
    4. **Model Building** — Feed-forward ANN (Dense + BatchNorm + Dropout layers)
    5. **Hyperparameter Tuning** — Optuna (20 trials) optimizing layers, units, dropout, learning rate
    6. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
    7. **Deployment** — This Streamlit app, with webcam + image upload prediction
    """)


# ---------------------------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------------------------
elif page == "Prediction":
    st.markdown('<p class="main-header">Try It Yourself</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a photo or use your webcam to detect a hand gesture</p>', unsafe_allow_html=True)
    st.divider()

    with st.expander("ℹ️ User Instructions", expanded=False):
        st.markdown("""
        - Make sure your **entire hand** is visible in the frame.
        - Use good lighting for best landmark detection accuracy.
        - Hold one of the 8 supported gestures: 👍 👎 ✋ ✊ ✌️ 👌 ☝️ 🤙
        - The model shows a **confidence score** for its prediction — low confidence
          may mean the gesture is ambiguous or partially out of frame.
        """)

    predictor = load_predictor()

    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Webcam"])

    def show_result(image_bgr):
        with st.spinner("Detecting hand and predicting gesture..."):
            result = predictor.predict(image_bgr)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB),
                      caption="Detected Landmarks", use_container_width=True)

        with col2:
            if not result["success"]:
                st.markdown(f"""
                <div class="warn-box">
                    <h3>⚠️ No Hand Detected</h3>
                    <p>{result['message']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                emoji = GESTURE_EMOJIS.get(result["gesture"], "🖐️")
                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size: 3rem;">{emoji}</div>
                    <h2>{result['gesture'].replace('_', ' ').title()}</h2>
                    <h4>Confidence: {result['confidence']*100:.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)

                log_prediction(result["gesture"], result["confidence"])

                st.write("")
                st.write("**Full Probability Breakdown:**")
                prob_df = pd.DataFrame({
                    "Gesture": list(result["probabilities"].keys()),
                    "Probability": list(result["probabilities"].values()),
                }).sort_values("Probability", ascending=True)

                fig = px.bar(prob_df, x="Probability", y="Gesture", orientation="h",
                             color="Probability", color_continuous_scale="Blues")
                fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            show_result(image_bgr)

    with tab2:
        camera_image = st.camera_input("Take a photo showing your gesture")
        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            show_result(image_bgr)

    if st.session_state.prediction_history:
        st.divider()
        with st.expander(f"🕒 Prediction History (last {len(st.session_state.prediction_history)})", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CONTROL CENTER PAGE — gestures trigger real, visible actions
# ---------------------------------------------------------------------------
elif page == "Control Center":
    st.markdown('<p class="main-header">🎮 Gesture Control Center</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Use gestures to control a slide deck and a mock media player — no touch required</p>', unsafe_allow_html=True)
    st.divider()

    with st.expander("ℹ️ Gesture → Action Map", expanded=True):
        action_cols = st.columns(4)
        for i, (gesture, action) in enumerate(GESTURE_ACTIONS.items()):
            with action_cols[i % 4]:
                st.markdown(f"**{GESTURE_EMOJIS[gesture]} {gesture.replace('_', ' ').title()}** → {action}")

    predictor = load_predictor()

    st.divider()
    col_cam, col_state = st.columns([1, 1.2])

    with col_cam:
        st.subheader("📷 Show a Gesture")
        control_image = st.camera_input("Capture a gesture to trigger an action", key="control_cam")

        detected_gesture = None
        if control_image is not None:
            image = Image.open(control_image).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            with st.spinner("Detecting gesture..."):
                result = predictor.predict(image_bgr)

            if result["success"]:
                detected_gesture = result["gesture"]
                action = apply_action(detected_gesture)
                log_prediction(detected_gesture, result["confidence"], action)
                st.success(f"{GESTURE_EMOJIS[detected_gesture]} Detected **{detected_gesture.replace('_',' ').title()}** → Action: **{action}**")
            else:
                st.warning("No hand detected — try again with your hand clearly in frame.")

    with col_state:
        st.subheader("🖥️ Live System State")

        # --- Mock slide deck ---
        st.markdown("**Slide Deck Control**")
        title, subtitle, color = SLIDES[st.session_state.slide_index]
        st.markdown(f"""
        <div style="background-color:{color}; color:white; border-radius:12px;
                    padding:2rem; text-align:center; margin-bottom:1rem;">
            <div style="font-size:0.9rem; opacity:0.85;">{title} of {len(SLIDES)}</div>
            <div style="font-size:1.4rem; font-weight:700; margin-top:0.5rem;">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress((st.session_state.slide_index + 1) / len(SLIDES))
        st.caption("👍 Thumbs Up = Next Slide  |  👎 Thumbs Down = Previous Slide")

        st.write("")

        # --- Mock media player ---
        st.markdown("**Media Player Control**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Status", "▶️ Playing" if st.session_state.is_playing else "⏸️ Paused")
        m2.metric("Volume", f"{'🔇 Muted' if st.session_state.is_muted else str(st.session_state.volume) + '%'}")
        m3.metric("Last Action", GESTURE_ACTIONS.get(detected_gesture, "-") if detected_gesture else "-")
        st.caption("✊ Fist = Play | ✋ Open Palm = Pause | ✌️ Peace = Vol Up | 🤙 Call Me = Vol Down | ☝️ Pointing = Mute")

    if st.button("🔄 Reset State"):
        st.session_state.slide_index = 0
        st.session_state.is_playing = False
        st.session_state.volume = 50
        st.session_state.is_muted = False
        st.rerun()

    if st.session_state.prediction_history:
        st.divider()
        with st.expander(f"🕒 Action History (last {len(st.session_state.prediction_history)})", expanded=True):
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True, hide_index=True)
