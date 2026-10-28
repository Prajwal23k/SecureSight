import streamlit as st
import requests
from PIL import Image
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SecureSight | AI Image Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body {
    font-family: 'Inter', sans-serif !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}


/* HERO */

.hero {
    background: linear-gradient(
        135deg,
        rgba(108,99,255,0.18),
        rgba(168,85,247,0.10)
    );

    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 22px;

    padding: 38px 30px;

    text-align: center;

    margin-bottom: 35px;
}

.hero-badge {
    display: inline-block;

    background: rgba(108,99,255,0.18);

    border: 1px solid rgba(108,99,255,0.4);

    color: #818cf8;

    padding: 6px 14px;

    border-radius: 30px;

    font-size: 12px;

    font-weight: 600;

    letter-spacing: 1px;

    margin-bottom: 15px;
}

.hero-title {
    font-size: 42px;

    font-weight: 800;

    margin-bottom: 12px;

    color: #e2e8f0;
}

.hero-sub {
    color: #94a3b8;

    font-size: 16px;

    max-width: 650px;

    margin: auto;

    line-height: 1.7;
}


/* RESULT */

.result {
    padding: 20px;

    border-radius: 16px;

    margin-bottom: 15px;
}

.result.fake {
    background: rgba(239,68,68,0.10);

    border: 1px solid rgba(239,68,68,0.45);
}

.result.real {
    background: rgba(34,197,94,0.10);

    border: 1px solid rgba(34,197,94,0.45);
}

.result-icon {
    font-size: 32px;
}

.result-label {
    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 1px;

    color: #94a3b8;

    margin-bottom: 3px;
}

.result-title {
    font-size: 22px;

    font-weight: 700;
}


/* SCORE CARDS */

.score {
    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.09);

    border-radius: 14px;

    padding: 16px 8px;

    text-align: center;

    min-height: 90px;
}

.score-label {
    color: #94a3b8;

    font-size: 10px;

    text-transform: uppercase;

    letter-spacing: 0.8px;
}

.score-number {
    font-size: 21px;

    font-weight: 700;

    margin-top: 8px;
}


/* FUSION */

.fusion {
    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.09);

    border-radius: 15px;

    padding: 18px;

    margin-top: 18px;
}

.fusion-title {
    color: #94a3b8;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 1px;

    margin-bottom: 10px;
}

.fusion-main {
    color: #e2e8f0;

    font-size: 15px;
}

.fusion-note {
    color: #94a3b8;

    font-size: 12px;

    margin-top: 10px;

    line-height: 1.5;
}


/* PIPELINE */

.pipeline {
    display: flex;

    justify-content: center;

    align-items: center;

    gap: 10px;

    flex-wrap: wrap;

    margin: 15px 0;
}

.pipeline-box {
    background: rgba(108,99,255,0.10);

    border: 1px solid rgba(108,99,255,0.25);

    border-radius: 10px;

    padding: 12px 16px;

    text-align: center;

    color: #cbd5e1;

    font-size: 12px;
}

.pipeline-arrow {
    color: #818cf8;

    font-size: 18px;
}


/* FOOTER */

.footer {
    text-align: center;

    margin-top: 35px;

    padding-top: 20px;

    border-top: 1px solid rgba(255,255,255,0.07);

    color: #64748b;

    font-size: 12px;
}

</style>
""")


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        🛡️ SecureSight · AI Detection Engine
    </div>

    <div class="hero-title">
        Is Your Image Real or AI-Generated?
    </div>

    <div class="hero-sub">
        Upload an image and SecureSight analyzes it using
        a CNN classifier and diffusion-based feature analysis
        to determine whether it is real or AI-generated.
    </div>

</div>
""")


# ============================================================
# COLUMNS
# ============================================================

left, right = st.columns(
    [1, 1.05],
    gap="large"
)


# ============================================================
# LEFT
# ============================================================

with left:

    st.markdown("### 📁 Upload Image")

    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        width, height = image.size

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"📐 {width} × {height}px   •   "
            f"🗂 {uploaded_file.type}   •   "
            f"💾 {uploaded_file.size / 1024:.1f} KB"
        )


    st.markdown("")


    analyze = st.button(
        "🔍  Analyse Image",
        disabled=uploaded_file is None,
        use_container_width=True
    )


# ============================================================
# RIGHT
# ============================================================

with right:

    st.markdown("### 📊 Detection Results")


    # --------------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------------

    if uploaded_file is None:

        st.html("""
        <div style="
            text-align:center;
            margin-top:100px;
            color:#64748b;
        ">

            <div style="font-size:60px;">
                🖼️
            </div>

            <div style="
                font-size:14px;
                line-height:1.7;
            ">
                Upload an image and click<br>
                <strong>Analyse Image</strong><br>
                to see the detection result.
            </div>

        </div>
        """)


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    elif analyze:

        progress = st.progress(
            0,
            text="Initializing SecureSight..."
        )


        progress.progress(
            20,
            text="Pre-processing image..."
        )


        progress.progress(
            40,
            text="Running CNN classifier..."
        )


        progress.progress(
            70,
            text="Running diffusion analysis..."
        )


        uploaded_file.seek(0)


        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",

                files={
                    "file": (
                        uploaded_file.name,

                        uploaded_file.read(),

                        uploaded_file.type
                    )
                },

                timeout=120
            )


        except requests.exceptions.ConnectionError:

            response = None


        except requests.exceptions.Timeout:

            response = "timeout"


        progress.progress(
            95,
            text="Combining model predictions..."
        )


        time.sleep(0.3)


        progress.progress(
            100,
            text="Analysis complete!"
        )


        time.sleep(0.3)


        progress.empty()


        # ----------------------------------------------------
        # BACKEND NOT RUNNING
        # ----------------------------------------------------

        if response is None:

            st.error(
                "Backend is not running."
            )

            st.code(
                "python -m uvicorn main:app --reload"
            )


        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        elif response == "timeout":

            st.error(
                "The analysis took too long. "
                "Please try again."
            )


        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        elif response.status_code != 200:

            st.error(
                f"Backend Error: {response.status_code}"
            )

            st.code(
                response.text
            )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        else:

            result = response.json()


            prediction = result.get(
                "prediction",
                "Unknown"
            )


            confidence = float(
                result.get(
                    "confidence",
                    0
                )
            )


            cnn_score = float(
                result.get(
                    "cnn_score",
                    0
                )
            )


            diffusion_score = float(
                result.get(
                    "diffusion_score",
                    0
                )
            )


            fusion_score = float(
                result.get(
                    "fusion_score",
                    0
                )
            )


            cnn_weight = float(
                result.get(
                    "cnn_weight",
                    0.6
                )
            )


            diffusion_weight = float(
                result.get(
                    "diffusion_weight",
                    0.4
                )
            )


            is_fake = (
                prediction.lower()
                == "fake"
            )


            # ------------------------------------------------
            # VERDICT
            # ------------------------------------------------

            if is_fake:

                icon = "🚫"

                verdict = "AI-Generated Image"

                css_class = "fake"

                title_color = "#f87171"

            else:

                icon = "✅"

                verdict = "Authentic / Real Image"

                css_class = "real"

                title_color = "#4ade80"


            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            st.html(
                f"""
                <div class="result {css_class}">

                    <div style="
                        display:flex;
                        align-items:center;
                        gap:14px;
                    ">

                        <div class="result-icon">
                            {icon}
                        </div>

                        <div>

                            <div class="result-label">
                                Final Verdict
                            </div>

                            <div class="result-title"
                                 style="color:{title_color};">

                                {verdict}

                            </div>

                        </div>

                    </div>

                </div>
                """
            )


            # ------------------------------------------------
            # SCORE COLOR FUNCTION
            # ------------------------------------------------

            def get_color(score):

                if score >= 0.75:

                    return "#f87171"

                elif score >= 0.50:

                    return "#fb923c"

                else:

                    return "#4ade80"


            # ------------------------------------------------
            # SCORE CARDS
            # ------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.html(
                    f"""
                    <div class="score">

                        <div class="score-label">
                            Confidence
                        </div>

                        <div class="score-number"
                             style="
                             color:{get_color(confidence)};
                             ">

                            {confidence:.1%}

                        </div>

                    </div>
                    """
                )


            with c2:

                st.html(
                    f"""
                    <div class="score">

                        <div class="score-label">
                            CNN Score
                        </div>

                        <div class="score-number"
                             style="
                             color:{get_color(cnn_score)};
                             ">

                            {cnn_score:.1%}

                        </div>

                    </div>
                    """
                )


            with c3:

                st.html(
                    f"""
                    <div class="score">

                        <div class="score-label">
                            Diffusion
                        </div>

                        <div class="score-number"
                             style="
                             color:{get_color(diffusion_score)};
                             ">

                            {diffusion_score:.1%}

                        </div>

                    </div>
                    """
                )


            with c4:

                st.html(
                    f"""
                    <div class="score">

                        <div class="score-label">
                            Fusion
                        </div>

                        <div class="score-number"
                             style="
                             color:{get_color(fusion_score)};
                             ">

                            {fusion_score:.1%}

                        </div>

                    </div>
                    """
                )


            # ------------------------------------------------
            # FUSION INFORMATION
            # ------------------------------------------------

            st.html(
                f"""
                <div class="fusion">

                    <div class="fusion-title">
                        Model Fusion
                    </div>

                    <div class="fusion-main">

                        CNN contribution:
                        <strong>
                            {cnn_weight:.0%}
                        </strong>

                        &nbsp;&nbsp;+

                        Diffusion contribution:
                        <strong>
                            {diffusion_weight:.0%}
                        </strong>

                    </div>

                    <div class="fusion-note">

                        Final prediction is generated by
                        combining the CNN and diffusion
                        model probabilities using weighted fusion.

                    </div>

                </div>
                """
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("")


with st.expander(
    "⚙️ How SecureSight Works"
):

    st.html("""
    <div class="pipeline">

        <div class="pipeline-box">
            📷<br>
            Image Upload
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-box">
            🧠<br>
            CNN
        </div>

        <div class="pipeline-arrow">
            +
        </div>

        <div class="pipeline-box">
            🌫️<br>
            Diffusion
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-box">
            ⚖️<br>
            60 : 40 Fusion
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-box">
            🛡️<br>
            Final Result
        </div>

    </div>
    """)


    st.markdown("")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown("""
        **🧠 CNN Classifier**

        Analyzes visual patterns in the
        uploaded image and produces
        a fake-image probability.
        """)


    with col2:

        st.markdown("""
        **🌫️ Diffusion Analysis**

        Analyzes noise prediction patterns
        across multiple diffusion timesteps.
        """)


    with col3:

        st.markdown("""
        **⚖️ Weighted Fusion**

        Combines the CNN and diffusion
        probabilities using **60% CNN
        + 40% Diffusion**.
        """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    SecureSight · AI Image Detection System ·
    CNN + Diffusion Model ·
    VESIT Capstone Project 2026

</div>
""")