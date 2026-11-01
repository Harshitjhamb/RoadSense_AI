import streamlit as st

st.set_page_config(
    page_title="Realtime Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Realtime Road Damage Detection")

st.info("🚧 **Coming soon**")

st.markdown(
    """
    This feature is currently under development.

    We will first evaluate the model in a **lab evaluation** setting
    (see the evaluation notebook in `training/`), and once the results are
    validated, realtime detection will be added to this page.

    In the meantime, please try:
    - 🖼️ **Image Detection** – upload a road image to detect damage
    - 🎞️ **Video Detection** – upload a video for frame-by-frame detection
    """
)
