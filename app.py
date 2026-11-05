"""Interface web locale pour classifier une image d’objet routier."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import CLASSES, MODELS_DIR
from src.predict import load_model, predict_image


st.set_page_config(
    page_title="RoadLens — Classification routière",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      .stApp { background: radial-gradient(ellipse at 10% 0%, #e8f4ff 0, transparent 38%), #f5f7fb; }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #10233f; }
      .hero { padding: 2rem 2.2rem; border-radius: 24px; color: white;
        background: linear-gradient(120deg, #10233f 0%, #174e75 58%, #19a7a1 100%);
        box-shadow: 0 16px 40px rgba(16, 35, 63, .18); margin-bottom: 1.5rem; }
      .hero h1 { color: white; margin: .25rem 0 .45rem; font-size: 2.35rem; }
      .hero p { color: #d7ecf6; margin: 0; font-size: 1.04rem; }
      .eyebrow { color: #7de0d4; text-transform: uppercase; letter-spacing: .14em; font-weight: 700; font-size: .76rem; }
      .result-card { padding: 1.25rem 1.5rem; border-radius: 18px; background: white;
        border: 1px solid #e4eaf2; box-shadow: 0 10px 28px rgba(25, 45, 70, .07); }
      .result-label { color: #10233f; font: 800 2rem 'Manrope', sans-serif; text-transform: capitalize; }
      .result-caption { color: #62728a; font-size: .92rem; }
      .small-note { color: #66758a; font-size: .86rem; }
      [data-testid="stFileUploader"] { background: white; border-radius: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Chargement du modèle entraîné…")
def get_model():
    return load_model()


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Projet PACTE · Machine Learning</div>
      <h1>RoadLens</h1>
      <p>Déposez une image d’objet routier et obtenez une prédiction en quelques secondes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## À propos")
    st.write("Classifieur SVM entraîné sur des objets recadrés de Pascal VOC 2007.")
    st.markdown("### Classes reconnues")
    st.write(" · ".join(label.capitalize() for label in CLASSES))
    st.divider()
    st.caption("La confiance est une estimation du modèle, pas une garantie de fiabilité.")

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("1 · Choisir une image")
    uploaded_file = st.file_uploader(
        "Formats acceptés : JPG, JPEG, PNG ou WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )
    st.markdown(
        '<p class="small-note">Conseil : choisissez une image où un seul objet est visible et occupe une bonne partie du cadre.</p>',
        unsafe_allow_html=True,
    )

with right:
    st.subheader("2 · Résultat")
    if uploaded_file is None:
        st.info("L’aperçu et la prédiction apparaîtront ici après l’import de votre image.")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)

            model_path = MODELS_DIR / "best_model.joblib"
            if not model_path.exists():
                st.error("Le modèle n’est pas encore entraîné. Lancez d’abord le pipeline du projet : python -m scripts.run_all")
            elif st.button("Classifier cette image", type="primary", use_container_width=True):
                suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    image.save(temp_path, format="JPEG" if suffix in {".jpg", ".jpeg"} else "PNG")
                try:
                    with st.spinner("Analyse des caractéristiques visuelles…"):
                        model = get_model()
                        label, confidence, scores = predict_image(temp_path, model=model)
                finally:
                    temp_path.unlink(missing_ok=True)

                st.markdown(
                    f'<div class="result-card"><div class="result-caption">Classe prédite</div>'
                    f'<div class="result-label">{label}</div>'
                    f'<div class="result-caption">Confiance estimée : <strong>{confidence * 100:.1f} %</strong></div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(max(0.0, min(1.0, confidence)))
                scores_df = pd.DataFrame(
                    {"Classe": [name.capitalize() for name in scores], "Score": list(scores.values())}
                ).sort_values("Score", ascending=True)
                st.markdown("#### Scores par classe")
                st.bar_chart(scores_df.set_index("Classe"), horizontal=True, x_label="Score estimé", y_label="Classe")
                st.caption("Le modèle classe l’image entière fournie : il ne détecte pas automatiquement plusieurs objets ni leurs positions.")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"Impossible de lire cette image : {exc}")
        except FileNotFoundError as exc:
            st.error(str(exc))
