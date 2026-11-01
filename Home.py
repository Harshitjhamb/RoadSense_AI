import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection | ML Semester Project",
    page_icon="🛣️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a11cb 100%);
        border-radius: 18px; padding: 2.2rem 1.8rem; color: #fff; text-align: center;
        box-shadow: 0 8px 24px rgba(42,82,152,.35); margin-bottom: 1.2rem;
    }
    .hero h1 { color:#fff; font-size: 2.2rem; margin:0 0 .4rem 0; }
    .hero p  { color:#e8eefc; margin:0; font-size:1.05rem; }
    .badge {
        display:inline-block; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
        padding:.25rem .8rem; border-radius:999px; font-size:.85rem; margin-bottom:.9rem;
    }
    .card {
        background:#fff; border-radius:14px; padding:1.1rem 1.2rem; height:100%;
        border:1px solid #e6e9f2; box-shadow:0 3px 12px rgba(0,0,0,.06);
    }
    .card h4 { margin:0 0 .4rem 0; color:#1e3c72; }
    .card p  { margin:0; color:#444; font-size:.95rem; }
    .chip {
        display:inline-block; background:#eef2ff; color:#2a3f9e; border-radius:10px;
        padding:.35rem .8rem; margin:.2rem .15rem; font-weight:600; font-size:.9rem;
    }
    .member {
        text-align:center; background:linear-gradient(180deg,#f7f8ff,#eef1ff);
        border:1px solid #dfe3f7; border-radius:14px; padding:.9rem .3rem; font-weight:600; color:#1e3c72;
    }
    .avatar {
        width:46px; height:46px; line-height:46px; border-radius:50%; margin:0 auto .4rem auto;
        background:linear-gradient(135deg,#2a5298,#6a11cb); color:#fff; font-size:1.2rem;
    }
    .link-pill {
        display:block; text-align:center; padding:.7rem; border-radius:12px;
        background:#f4f6fb; border:1px dashed #b8c0e0; color:#1e3c72; font-weight:600;
    }
    .link-pill small { display:block; color:#7a829e; font-weight:400; }
    </style>

    <div class="hero">
        <div class="badge">🎓 Machine Learning · Semester Project</div>
        <h1>🛣️ Road Damage Detection</h1>
        <p>Pothole &amp; crack detection on images and videos using YOLOv8,<br>
        trained on Indian road data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)



st.markdown("### 🔍 About the Project")
st.write(
    "This Machine Learning semester project detects and classifies road damage to "
    "support road safety and infrastructure maintenance. It uses a **YOLOv8 (small)** "
    "deep learning model trained on the **India** subset of the "
    "Crowdsensing-based Road Damage Detection Challenge 2022 (CRDDC2022) dataset."
)

st.markdown("### 🚧 Damage Types Detected")
st.markdown(
    """
    <span class="chip">↕ Longitudinal Crack</span>
    <span class="chip">↔ Transverse Crack</span>
    <span class="chip">🕸 Alligator Crack</span>
    <span class="chip">🕳 Pothole</span>
    """,
    unsafe_allow_html=True,
)

st.markdown("### ✨ Features")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        <div class="card">
            <h4>🖼️ Image Pothole Detection</h4>
            <p>Upload a road photo and get potholes and cracks highlighted
            with bounding boxes and confidence scores.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="card">
            <h4>🎞️ Video Detection</h4>
            <p>Upload a road video and get every frame analysed, with an
            annotated output video to download.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.caption("👈 Choose Image Detection, Video Detection or Realtime Detection from the sidebar.")

st.markdown("### 👥 Team")
team = ["Harshit Jhamb","Krish Karandikar", "Abhiram", "Gufran Alam", "Devam Kharbandha"]
cols = st.columns(5)
for col, name in zip(cols, team):
    with col:
        st.markdown(
            f'<div class="member"><div class="avatar">{name[0]}</div>{name}</div>',
            unsafe_allow_html=True,
        )

st.markdown("### 🔗 Documentation & Links")
l1, l2 = st.columns(2)
with l1:
        st.markdown(
    '''
    <div class="link-pill">
        <a href="https://github.com/Harshitjhamb/RoadSense_AI" target="_blank">GitHub</a>
        <small>View Source Code ↗</small>
    </div>
    ''',
    unsafe_allow_html=True
)
with l2:
   st.markdown(
    '''
    <div class="link-pill">
        <a href="https://docs.google.com/document/d/1q8E7ipjZ3ycM4Mclm2cLY9mwE1X_wjHhZg6n4aKdSII/edit?tab=t.0"
           target="_blank">
            Project Report ↗
        </a>
         <small>View Project Documentation ↗</small>
    </div>
    ''',
    unsafe_allow_html=True
)

st.divider()
st.caption("Machine Learning Semester Project · Team of 5 · Built with YOLOv8 and Streamlit")

