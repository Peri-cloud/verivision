"""VeriVision AI - Streamlit front end.

The UI only wires cached detector singletons into the pure functions in
analysis.py; all interpretation lives in utils/result_formatting.py so the
science and the presentation can never drift apart.

Run with:  python -m streamlit run app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make project-root imports robust regardless of where Streamlit is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import config
from analysis import run_full_analysis
from utils.image_validation import ImageValidationError, validate_image_bytes
from utils.preprocessing import image_stats
from utils.result_formatting import VERDICT_STYLES, Verdict, format_duration

st.set_page_config(
    page_title="VeriVision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODE_AI = "AI-Generated Image Detection"
MODE_DEEPFAKE = "Deepfake Face Detection"
MODE_BOTH = "Both Detectors"
ALL_MODES = [MODE_AI, MODE_DEEPFAKE, MODE_BOTH]


# ------------------------------------------------------------------ models
@st.cache_resource(show_spinner=False)
def get_ai_detector():
    from models.ai_image_detector import AIImageDetector

    return AIImageDetector().load()


@st.cache_resource(show_spinner=False)
def get_face_detector():
    from models.face_detector import FaceDetector

    return FaceDetector().load()


@st.cache_resource(show_spinner=False)
def get_deepfake_detector():
    from models.deepfake_detector import DeepfakeDetector

    return DeepfakeDetector().load()


# ------------------------------------------------------------------- theming
TONE_CSS = {
    "positive": ("#e6f4ea", "#137333"),
    "negative": ("#fce8e6", "#c5221f"),
    "warning": ("#fef7e0", "#b26a00"),
    "neutral": ("#f1f3f4", "#3c4043"),
}


def verdict_badge(verdict: Verdict, subtitle: str = "") -> str:
    tone = VERDICT_STYLES.get(verdict, {}).get("tone", "neutral")
    bg, fg = TONE_CSS[tone]
    sub = f'<div style="font-size:0.82rem;margin-top:2px;opacity:.85;">{subtitle}</div>' if subtitle else ""
    return (
        f'<div style="background:{bg};color:{fg};padding:0.6rem 0.9rem;'
        f'border-radius:0.5rem;border:1px solid {fg}33;">'
        f'<span style="font-size:1.05rem;font-weight:700;">{verdict.value}</span>'
        f"{sub}</div>"
    )


def score_bar(score: float | None, caption: str):
    if score is None:
        return
    st.progress(min(max(score, 0.0), 1.0), text=f"Score: {score:.4f}")
    st.caption(caption)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🔍 VeriVision AI")
    st.caption(config.APP_SUBTITLE)

    st.subheader("Analysis mode")
    mode = st.radio("Choose one", ALL_MODES, index=2, label_visibility="collapsed")

    with st.expander("Models & versions"):
        st.markdown(
            f"**A · AI-image detector**\n"
            f"`{config.AI_IMAGE_MODEL_ID}`\n\n"
            f"SwinV2, {config.AI_IMAGE_SIZE_PX}px input, Apache-2.0. Detects "
            "whole-image AI synthesis. Not a face-manipulation detector.\n\n"
            f"**B · Deepfake face detector**\n"
            f"`{config.DEEPFAKE_MODEL_ID}`\n\n"
            f"ViT-B/16, {config.DEEPFAKE_SIZE_PX}px input, Apache-2.0. Detects "
            "facial manipulation; applied per detected face only.\n\n"
            f"**Face detector:** OpenCV YuNet (Apache-2.0) with Haar-cascade "
            "fallback.\n\n"
            f"**Weights location:** `{config.WEIGHTS_DIR}`"
        )
    with st.expander("Documented limitations"):
        st.markdown(
            "- Scores are **uncalibrated** softmax outputs, not probabilities "
            "of ground truth.\n"
            "- Both models were trained years ago; newer generators can evade "
            "them (concept drift).\n"
            "- The deepfake model is face-specific and is never applied to "
            "face-free images.\n"
            "- This tool does **not** identify people and cannot tell who "
            "created or edited an image.\n"
            "- Single-image analysis only; no video support in this version."
        )
    st.info(config.DISCLAIMER, icon="⚠️")

# --------------------------------------------------------------------- hero
st.title("🔍 VeriVision AI")
st.markdown(
    "Upload a photograph to check it with **independent detectors** for "
    "AI-generated imagery and for deepfake face manipulation. Every result "
    "shows its raw model score, its limits, and what the score does *not* mean."
)

# ------------------------------------------------------------------- upload
upload = st.file_uploader(
    "Upload a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    key="upload",
    help="Files are validated by content, not by extension.",
)

if upload is None:
    st.info("⬅️ Upload an image to begin, or try one from `assets/sample_images/`.", icon="🖼️")
    st.stop()

image_bytes = upload.getvalue()
try:
    image, upload_info = validate_image_bytes(image_bytes)
except ImageValidationError as exc:
    st.error(exc.user_message, icon="🚫")
    if exc.detail:
        with st.expander("Technical error detail"):
            st.code(exc.detail)
    if st.button("Analyze another image", type="secondary"):
        del st.session_state["upload"]
        st.rerun()
    st.stop()

# ------------------------------------------------------------ preview panel
preview_col, meta_col = st.columns([3, 2])
with preview_col:
    st.image(image, use_container_width=True, caption="Uploaded image")
with meta_col:
    st.subheader("Image details")
    stats = image_stats(image)
    st.markdown(
        f"- **Dimensions:** {stats['width']} × {stats['height']} px\n"
        f"- **Megapixels:** {upload_info['megapixels']}\n"
        f"- **True format:** {upload_info['declared_format']}\n"
        f"- **Colour mode:** {upload_info['original_mode']} → analysed as RGB\n"
        f"- **File size:** {upload_info['file_bytes'] / 1024:.0f} KB"
    )
    st.caption(f"Selected mode: **{mode}**")

st.divider()

# ------------------------------------------------------------------ analyse
run_clicked = st.button("🔬 Analyze Image", type="primary", use_container_width=True)

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if run_clicked:
    st.session_state.last_analysis = {"fingerprint": hash(image_bytes), "mode": mode}

if st.session_state.last_analysis is None or st.session_state.last_analysis["fingerprint"] != hash(image_bytes):
    st.info("Press **Analyze Image** to run the selected detector(s).")
    st.stop()

analysis_mode = st.session_state.last_analysis["mode"]

# Load only the models the selected mode needs, with progress feedback.
detectors_needed = []
if analysis_mode in (MODE_AI, MODE_BOTH):
    detectors_needed.append("AI-image detector")
if analysis_mode in (MODE_DEEPFAKE, MODE_BOTH):
    detectors_needed.append("face detector + deepfake detector")

ai_detector = face_detector = deepfake_detector = None
try:
    with st.spinner(
        f"Loading detector weights (first run downloads them once, ~0.8 GB): "
        f"{', '.join(detectors_needed)}…"
    ):
        load_start = time.perf_counter()
        if analysis_mode in (MODE_AI, MODE_BOTH):
            ai_detector = get_ai_detector()
        if analysis_mode in (MODE_DEEPFAKE, MODE_BOTH):
            face_detector = get_face_detector()
            deepfake_detector = get_deepfake_detector()
    st.caption(
        f"Models ready in {format_duration((time.perf_counter() - load_start) * 1000)} "
        f"(cached after first load)."
    )
except Exception as exc:  # ModelLoadError etc.
    st.error(
        f"**The detector could not be loaded.** {exc}", icon="❌"
    )
    st.stop()

with st.spinner("Running inference…"):
    results = run_full_analysis(
        image,
        ai_detector=ai_detector,
        face_detector=face_detector,
        deepfake_detector=deepfake_detector,
        mode=analysis_mode,
    )

# ------------------------------------------------------------------ results
st.subheader("Results")


def render_ai_result(res: dict):
    with st.container(border=True):
        st.markdown("#### 🤖 AI-Generated Image Detector")
        if res["verdict"] is Verdict.ERROR:
            st.error(f"The detector could not produce a result. {res['error_message']}", icon="❌")
        else:
            note = VERDICT_STYLES[res["verdict"]]["note"]
            st.markdown(
                verdict_badge(res["verdict"], note),
                unsafe_allow_html=True,
            )
            score_bar(
                res["fake_score"],
                res["score_description"],
            )
            if res["used_uncertainty_band"]:
                st.warning(
                    "The score is inside the documented uncertainty band "
                    f"(±{config.UNCERTAINTY_BAND} around {config.DECISION_BOUNDARY}), "
                    "so the image is reported as inconclusive instead of being forced "
                    "into a binary verdict.",
                    icon="⚠️",
                )
        for warning in res.get("warnings", []):
            st.caption(f"ℹ️ {warning}")
        with st.expander("Technical details"):
            st.json({k: v for k, v in res.items() if k not in ("score_description",)}, expanded=False)
            st.caption(f"Score meaning: {res['score_description']}")


def render_deepfake_result(res: dict):
    with st.container(border=True):
        st.markdown("#### 👤 Deepfake Face Detector")
        summary = res["summary"]

        if res.get("annotated_image") is not None:
            st.image(
                res["annotated_image"],
                use_container_width=True,
                caption=f"{res['face_count']} face(s) analysed",
            )
        st.markdown(
            verdict_badge(
                summary["verdict"],
                summary["narrative"],
            ),
            unsafe_allow_html=True,
        )

        if not res["faces"]:
            st.caption(
                "No deepfake score is reported: the deepfake model is trained "
                "for facial manipulation and was not applied."
            )
        else:
            cols = st.columns(min(3, max(1, len(res["faces"]))))
            for i, face in enumerate(res["faces"]):
                with cols[i % len(cols)], st.container(border=True):
                    st.image(face["crop"], use_container_width=True, caption=f"Face {face['index']}")
                    if face["verdict"] is Verdict.ERROR:
                        st.error("Inference failed for this face.", icon="❌")
                        st.caption(face.get("error_message", ""))
                        continue
                    st.markdown(
                        verdict_badge(face["verdict"]),
                        unsafe_allow_html=True,
                    )
                    score_bar(face["fake_score"], face["score_description"])

        for limitation in res.get("limitations", []):
            st.caption(f"ℹ️ {limitation}")
        with st.expander("Technical details"):
            tech = {
                "face_count": res["face_count"],
                "summary": {"verdict": summary["verdict"].value, "counts": summary["counts"]},
                "total_ms": res["total_ms"],
                "faces": [
                    {
                        "index": f["index"],
                        "box": f["box"],
                        "verdict": f["verdict"].value,
                        "fake_score": f["fake_score"],
                        "probabilities": f.get("probabilities", {}),
                        "model_id": f.get("model_id", config.DEEPFAKE_MODEL_ID),
                        "device": f.get("device"),
                        "inference_ms": f.get("inference_ms"),
                    }
                    for f in res["faces"]
                ],
                "limitations": res.get("limitations", []),
            }
            st.json(tech, expanded=False)
            st.caption(
                "Per-face scores are raw softmax outputs of the 'Fake' class - "
                "not calibrated probabilities."
            )


if results["ai_image"] is not None:
    render_ai_result(results["ai_image"])
if results["deepfake"] is not None:
    render_deepfake_result(results["deepfake"])

st.caption(
    f"Processing time: {format_duration(results['total_ms'])} total · "
    f"mode: {results['mode']} · scores are uncalibrated model outputs."
)

# ------------------------------------------------------------- another image
st.divider()
if st.button("🔄 Analyze another image", type="secondary", use_container_width=True):
    st.session_state.last_analysis = None
    if "file_uploader" in st.session_state:
        del st.session_state["upload"]
    st.rerun()

st.info(config.DISCLAIMER, icon="⚠️")
