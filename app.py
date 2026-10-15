from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing import image as keras_image


st.set_page_config(
    page_title="Wildfire Detection",
    page_icon="🔥",
    layout="centered",
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "4 cnn models"
FUSION_DIR = PROJECT_DIR / "fusion model"
SENTIMENT_MODEL_PATH = FUSION_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = FUSION_DIR / "vectorizer.pkl"

# Names and checkpoints match the four original training/prediction notebooks.
MODELS = {
    "Custom CNN": MODEL_DIR / "wildfire_classification_model_cnn.h5",
    "ResNet50": MODEL_DIR / "wildfire_classification_model_model3.h5",
    "MobileNetV2": MODEL_DIR / "wildfire_classification_model_model2.h5",
    "Attention CNN": MODEL_DIR / "wildfire_classification_model_model4.h5",
}


st.markdown(
    """
    <style>
        :root {
            --espresso: #2C1B15;
            --chocolate: #4A3025;
            --caramel: #C68B59;
            --cream: #F5E9D8;
            --beige: #C9B29C;
            --terracotta: #B85C38;
            --card-border: rgba(198, 139, 89, 0.28);
        }

        .stApp,
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 92% 8%, rgba(198, 139, 89, 0.10), transparent 30%),
                var(--espresso);
            color: var(--cream);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 840px;
            padding-top: 2.5rem;
            padding-bottom: 3.5rem;
        }

        h1, h2, h3, h4, h5, h6,
        p, label, .stMarkdown, [data-testid="stMetricLabel"] {
            color: var(--cream);
        }
        .stCaption, [data-testid="stCaptionContainer"], small {
            color: var(--beige) !important;
        }

        .hero {
            padding: 1.75rem 1.9rem;
            border: 1px solid var(--card-border);
            border-radius: 18px;
            background: linear-gradient(135deg, #4A3025 0%, #38231B 100%);
            color: var(--cream);
            margin-bottom: 1.75rem;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.20);
        }
        .hero h1 {
            margin: 0 0 .4rem 0;
            color: var(--cream);
            font-size: 2.05rem;
            letter-spacing: -0.025em;
        }
        .hero p { margin: 0; color: var(--beige); line-height: 1.6; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
            padding: .35rem;
            border: 1px solid var(--card-border);
            border-radius: 13px;
            background: var(--chocolate);
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.8rem;
            padding: 0 1rem;
            border-radius: 9px;
            color: var(--beige);
            transition: background .18s ease, color .18s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(198, 139, 89, .14);
            color: var(--cream);
        }
        .stTabs [aria-selected="true"] {
            background: var(--caramel) !important;
            color: var(--espresso) !important;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-highlight"] { background: transparent; }
        .stTabs [data-baseweb="tab-border"] { display: none; }

        /* Inputs, dropdowns and uploaders */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stFileUploaderDropzone"] {
            border: 1px solid var(--card-border) !important;
            border-radius: 12px !important;
            background: var(--chocolate) !important;
            color: var(--cream) !important;
            transition: border-color .18s ease, box-shadow .18s ease;
        }
        [data-testid="stSelectbox"] > div > div:hover,
        [data-testid="stTextInput"] input:hover,
        [data-testid="stTextArea"] textarea:hover,
        [data-testid="stNumberInput"] input:hover,
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--caramel) !important;
            box-shadow: 0 0 0 2px rgba(198, 139, 89, .10);
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {
            color: var(--cream) !important;
        }
        input::placeholder, textarea::placeholder {
            color: var(--beige) !important;
            opacity: .75;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--beige) !important;
        }
        [data-testid="stFileUploaderFile"] {
            border-radius: 10px;
            background: rgba(74, 48, 37, .72);
        }
        [data-baseweb="popover"] ul,
        [data-baseweb="menu"] {
            border: 1px solid var(--card-border);
            background: var(--chocolate) !important;
        }
        [role="option"] { color: var(--cream) !important; }
        [role="option"]:hover,
        [aria-selected="true"][role="option"] {
            background: rgba(198, 139, 89, .20) !important;
        }

        /* Buttons */
        div.stButton > button {
            width: 100%;
            min-height: 2.8rem;
            border: 1px solid var(--caramel);
            border-radius: 11px;
            background: var(--caramel);
            color: var(--espresso);
            font-weight: 700;
            transition: transform .16s ease, background .16s ease, box-shadow .16s ease;
        }
        div.stButton > button:hover:not(:disabled) {
            border-color: #D39A69;
            background: #D39A69;
            color: var(--espresso);
            box-shadow: 0 8px 20px rgba(198, 139, 89, .18);
            transform: translateY(-1px);
        }
        div.stButton > button:active:not(:disabled) { transform: translateY(0); }
        div.stButton > button:disabled {
            border-color: rgba(201, 178, 156, .20);
            background: rgba(201, 178, 156, .12);
            color: rgba(201, 178, 156, .55);
        }

        /* Cards and result panels */
        [data-testid="stMetric"],
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--card-border);
            border-radius: 14px;
            background: var(--chocolate);
            padding: 1rem 1.15rem;
        }
        [data-testid="stMetricValue"] { color: var(--cream); }
        [data-testid="stAlert"] {
            border-radius: 12px;
            color: var(--cream);
        }
        [data-testid="stAlert"] p { color: inherit; }
        [data-testid="stAlert"][data-baseweb="notification"] {
            border-color: var(--card-border);
            background: var(--chocolate);
        }
        .stAlert:has([data-testid="stNotificationContentError"]) {
            border-color: var(--terracotta) !important;
            background: rgba(184, 92, 56, .20) !important;
        }
        .stAlert:has([data-testid="stNotificationContentSuccess"]) {
            border-color: rgba(198, 139, 89, .65) !important;
            background: rgba(198, 139, 89, .14) !important;
        }
        [data-testid="stImage"] img {
            border: 1px solid var(--card-border);
            border-radius: 14px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, .18);
        }
        [data-testid="stImageCaption"] { color: var(--beige); }
        hr { border-color: rgba(201, 178, 156, .16); }

        /* Federated-learning research diagram */
        .fl-diagram {
            display: grid;
            grid-template-columns: repeat(5, minmax(90px, 1fr));
            gap: .65rem;
            align-items: center;
            margin: 1rem 0 1.5rem;
        }
        .fl-server {
            grid-column: 2 / 5;
            padding: 1rem;
            border: 1px solid var(--caramel);
            border-radius: 13px;
            background: var(--chocolate);
            color: var(--cream);
            text-align: center;
            box-shadow: 0 10px 25px rgba(0, 0, 0, .16);
        }
        .fl-flow {
            grid-column: 1 / -1;
            color: var(--caramel);
            text-align: center;
            font-size: .9rem;
            letter-spacing: .02em;
        }
        .fl-client {
            padding: .8rem .45rem;
            border: 1px solid var(--card-border);
            border-radius: 11px;
            background: rgba(74, 48, 37, .72);
            color: var(--cream);
            text-align: center;
            font-size: .9rem;
        }
        .fl-client span, .fl-server span {
            display: block;
            margin-top: .22rem;
            color: var(--beige);
            font-size: .78rem;
        }
        .process-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: .65rem;
            margin: .75rem 0 1.5rem;
        }
        .process-step {
            padding: .8rem;
            border-left: 3px solid var(--caramel);
            border-radius: 8px;
            background: var(--chocolate);
            color: var(--cream);
            font-size: .86rem;
        }
        .process-step strong { color: var(--caramel); }

        @media (max-width: 640px) {
            .block-container { padding: 1.25rem 1rem 2.5rem; }
            .hero { padding: 1.35rem 1.25rem; }
            .hero h1 { font-size: 1.7rem; }
            .stTabs [data-baseweb="tab"] { padding: 0 .65rem; }
            .fl-diagram { grid-template-columns: 1fr; }
            .fl-server, .fl-flow { grid-column: 1; }
            .process-row { grid-template-columns: 1fr; }
        }
    </style>
    <div class="hero">
        <h1>Wildfire Detection</h1>
        <p>Upload a satellite or landscape image and analyse it with a trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


available_models = {
    name: path for name, path in MODELS.items() if path.is_file()
}

if not available_models:
    st.error("No trained model checkpoints were found in the '4 cnn models' folder.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_selected_model(model_path: str):
    # compile=False is sufficient for inference and does not modify the checkpoint.
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_resource(show_spinner=False)
def load_sentiment_artifacts(model_path: str, vectorizer_path: str):
    """Load only the project's trusted, local research artifacts."""
    return joblib.load(model_path), joblib.load(vectorizer_path)


def preprocess_image(uploaded_image: Image.Image, model_name: str) -> np.ndarray:
    """Apply the preprocessing used in the corresponding original notebook."""
    rgb_image = uploaded_image.convert("RGB")
    resized_image = rgb_image.resize((224, 224))

    if model_name == "Custom CNN":
        image_array = np.array(resized_image) / 255.0
    else:
        image_array = keras_image.img_to_array(resized_image) / 255.0

    return np.expand_dims(image_array, axis=0)


def classify_probability(probability: float) -> tuple[str, float]:
    """Use the original notebooks' threshold and binary label mapping."""
    if probability > 0.5:
        return "Fire Detected", probability
    return "No Fire Detected", 1.0 - probability


def analyse_sentiment(tweet_text: str, sentiment_model, vectorizer) -> str:
    """Apply the saved TF-IDF transform and the notebook's label mapping."""
    tweet_vectorized = vectorizer.transform([tweet_text])
    predicted_label = sentiment_model.predict(tweet_vectorized)[0]
    return "Positive" if predicted_label == 1 else "Negative"


tab_image, tab_fusion, tab_federated = st.tabs(
    ["Image Detection", "Twitter Sentiment + Fusion", "Federated Learning"]
)


with tab_image:
    st.subheader("Image Detection")
    model_name = st.selectbox(
        "Select a model",
        list(available_models.keys()),
        key="image_model",
    )
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG",
        key="image_upload",
    )

    uploaded_image = None
    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file)
            uploaded_image.load()
            st.image(
                uploaded_image,
                caption=uploaded_file.name,
                use_container_width=True,
            )
        except (UnidentifiedImageError, OSError):
            st.error("The uploaded file could not be read as an image.")

    if st.button(
        "Analyse Image",
        type="primary",
        disabled=uploaded_image is None,
        key="analyse_image",
    ):
        try:
            with st.spinner(f"Analysing with {model_name}..."):
                model = load_selected_model(str(available_models[model_name]))
                model_input = preprocess_image(uploaded_image, model_name)
                probability = float(model.predict(model_input, verbose=0)[0][0])
                label, confidence = classify_probability(probability)

            if label == "Fire Detected":
                st.error(f"🔥 {label}")
            else:
                st.success(f"✅ {label}")

            st.metric("Confidence", f"{confidence:.2%}")
            st.caption(f"Model: {model_name}")
        except Exception as exc:
            st.error(f"The image could not be analysed: {exc}")


with tab_fusion:
    st.subheader("Twitter Sentiment + Multimodal Fusion")
    st.caption(
        "Experimental research demonstration using local model artifacts. "
        "It does not retrieve posts or send alerts by email."
    )

    if not SENTIMENT_MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
        st.error(
            "The sentiment model or TF-IDF vectorizer is missing from the "
            "'fusion model' folder."
        )
    else:
        tweet_text = st.text_area(
            "Enter a Tweet",
            placeholder="Enter wildfire-related text for sentiment analysis...",
            key="fusion_tweet",
        )
        fusion_model_name = st.selectbox(
            "Select an image model",
            list(available_models.keys()),
            key="fusion_image_model",
        )
        fusion_file = st.file_uploader(
            "Upload an image for fusion",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG and PNG",
            key="fusion_upload",
        )

        fusion_image = None
        if fusion_file is not None:
            try:
                fusion_image = Image.open(fusion_file)
                fusion_image.load()
                st.image(
                    fusion_image,
                    caption=fusion_file.name,
                    use_container_width=True,
                )
            except (UnidentifiedImageError, OSError):
                st.error("The uploaded file could not be read as an image.")

        if st.button(
            "Analyse Multimodal Input",
            type="primary",
            key="analyse_fusion",
        ):
            if not tweet_text.strip():
                st.warning("Enter a Tweet before running the analysis.")
            elif fusion_image is None:
                st.warning("Upload a valid image before running the analysis.")
            else:
                try:
                    with st.spinner("Analysing image and text..."):
                        sentiment_model, vectorizer = load_sentiment_artifacts(
                            str(SENTIMENT_MODEL_PATH),
                            str(VECTORIZER_PATH),
                        )
                        sentiment = analyse_sentiment(
                            tweet_text,
                            sentiment_model,
                            vectorizer,
                        )

                        image_model = load_selected_model(
                            str(available_models[fusion_model_name])
                        )
                        image_input = preprocess_image(
                            fusion_image,
                            fusion_model_name,
                        )
                        image_probability = float(
                            image_model.predict(image_input, verbose=0)[0][0]
                        )
                        image_label, image_confidence = classify_probability(
                            image_probability
                        )

                    fire_detected = image_probability > 0.5
                    alert_triggered = fire_detected and sentiment != "Positive"

                    result_col1, result_col2 = st.columns(2)
                    with result_col1:
                        st.metric("Image Classification", image_label)
                        st.caption(f"Confidence: {image_confidence:.2%}")
                    with result_col2:
                        st.metric("Tweet Sentiment", sentiment)
                        st.caption(f"Image model: {fusion_model_name}")

                    if alert_triggered:
                        st.error(
                            "🔥 Wildfire Alert! Fire was detected and the "
                            "tweet sentiment is Negative. Immediate attention "
                            "may be required."
                        )
                        st.markdown("### Final fusion decision: Wildfire Alert")
                    else:
                        st.success(
                            "No immediate threat detected by the original "
                            "fusion rule."
                        )
                        st.markdown(
                            "### Final fusion decision: No Immediate Threat"
                        )
                except Exception as exc:
                    st.error(f"The multimodal analysis could not be completed: {exc}")


with tab_federated:
    st.subheader("Federated Learning")
    st.caption(
        "Historical research experiment — this tab presents recorded notebook "
        "results and does not start a Flower server, train models, or perform "
        "live inference."
    )

    st.markdown(
        """
        Federated learning lets several data holders train a shared wildfire
        image classifier without pooling their image datasets in one place.
        In this experiment, each client trained the same two-class CNN on its
        own local train split, evaluated the received global parameters on its
        local validation split, and exchanged model parameters—not images—with
        a central Flower server.
        """
    )

    st.markdown("#### Verified experiment configuration")
    config_col1, config_col2, config_col3, config_col4 = st.columns(4)
    config_col1.metric("Clients", "5")
    config_col2.metric("Global rounds", "5")
    config_col3.metric("Local epochs", "3")
    config_col4.metric("Aggregation", "FedAvg")
    st.caption(
        "Server settings: fraction_fit=1.0, min_fit_clients=5, "
        "min_available_clients=5, address 127.0.0.1:8099."
    )

    st.markdown("#### Original client/server architecture")
    st.markdown(
        """
        <div class="fl-diagram">
            <div class="fl-server">
                Flower Server · FedAvg
                <span>Aggregates parameters and redistributes the global model</span>
            </div>
            <div class="fl-flow">↑ local model updates &nbsp;&nbsp; · &nbsp;&nbsp; ↓ global parameters</div>
            <div class="fl-client">Client 0<span>Local image data</span></div>
            <div class="fl-client">Client 1<span>Local image data</span></div>
            <div class="fl-client">Client 2<span>Local image data</span></div>
            <div class="fl-client">Client 3<span>Local image data</span></div>
            <div class="fl-client">Client 4<span>Local image data</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Training process")
    st.markdown(
        """
        <div class="process-row">
            <div class="process-step"><strong>1 · Distribute</strong><br>Server sends global parameters to all five clients.</div>
            <div class="process-step"><strong>2 · Train locally</strong><br>Each client trains for three epochs on local images.</div>
            <div class="process-step"><strong>3 · Aggregate</strong><br>Server combines all five updates using FedAvg.</div>
            <div class="process-step"><strong>4 · Evaluate</strong><br>Clients evaluate the aggregated parameters locally.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Historical recorded metrics")
    st.info(
        "These values were read from saved outputs in the original notebooks. "
        "They are not live measurements from the running Streamlit app."
    )

    distributed_loss = pd.DataFrame(
        {
            "Round": [1, 2, 3, 4, 5],
            "Distributed validation loss": [
                129.03023986816407,
                4.563878726959229,
                0.5064505994319916,
                0.522041630744934,
                0.6353179931640625,
            ],
        }
    ).set_index("Round")
    st.markdown("**Server-reported distributed validation loss**")
    st.line_chart(distributed_loss, color="#C68B59")
    st.caption(
        "The first-round loss is much larger than later rounds, so it "
        "dominates the linear chart scale. The recorded run completed five "
        "rounds in 161.85 seconds."
    )

    client_validation_accuracy = pd.DataFrame(
        {
            "Round": [1, 2, 3, 4, 5],
            "Client 0": [0.50, 0.87, 0.57, 0.65, 0.83],
            "Client 1": [0.50, 0.77, 0.62, 0.72, 0.79],
            "Client 2": [0.50, 0.87, 0.62, 0.65, 0.77],
            "Client 3": [0.50, 0.86, 0.58, 0.73, 0.81],
            "Client 4": [0.50, 0.87, 0.65, 0.69, 0.75],
        }
    ).set_index("Round")
    st.markdown("**Per-client validation accuracy recorded after each round**")
    st.line_chart(
        client_validation_accuracy,
        color=["#C68B59", "#B85C38", "#D9A86C", "#A96F4B", "#8F6A55"],
    )
    st.caption(
        "These are individual client validation accuracies from the five "
        "client notebook logs. The server did not define an accuracy "
        "aggregation function, so no global accuracy metric was recorded."
    )

    st.markdown("#### Available experiment artifacts")
    st.markdown(
        """
        - Five client notebooks: `Client 0.ipynb` through `Client 4.ipynb`
        - One Flower server notebook: `Server.ipynb`
        - Five final client checkpoints: `modelfinal_client_0.h5` through `modelfinal_client_4.h5`
        - One saved global checkpoint: `global_model.h5`
        """
    )
    st.warning(
        "The original dataset folders are referenced outside this project, "
        "and no separate CSV/JSON training-history export is present. The "
        "charts therefore use only metrics preserved in notebook outputs."
    )
