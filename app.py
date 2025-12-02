"""Streamlit UI for Arabic Fake News Detection.

A web interface for classifying Arabic news articles as real or fake.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

from src.feature_extraction import FeatureExtractor
from src.model_training import load_model


# Page configuration
st.set_page_config(
    page_title="Arabic Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for RTL Arabic text support
st.markdown("""
<style>
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-size: 1.2em;
        line-height: 1.8;
    }
    .result-real {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-fake {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load the trained model and feature extractor."""
    model_path = "best_model.joblib"
    extractor_path = "feature_extractor.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(extractor_path):
        return None, None
    
    model = load_model(model_path)
    extractor = FeatureExtractor.load(extractor_path)
    return model, extractor


@st.cache_data
def load_dataset_stats():
    """Load dataset statistics."""
    data_dir = "dataset"
    stats = {}
    
    for split in ["train", "validation", "test"]:
        path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            stats[split] = {
                "total": len(df),
                "real": (df["validity"] == 1).sum(),
                "fake": (df["validity"] == 0).sum(),
            }
    return stats


def predict_text(text: str, model, extractor) -> dict:
    """Predict if text is real or fake news."""
    features = extractor.transform([text])
    prediction = model.model.predict(features)[0]
    
    # Get probability if available
    if hasattr(model.model, "predict_proba"):
        proba = model.model.predict_proba(features)[0]
        confidence = max(proba)
        proba_fake, proba_real = proba[0], proba[1]
    else:
        # For SVM with decision_function
        decision = model.model.decision_function(features)[0]
        confidence = abs(decision)
        proba_fake = 1 / (1 + np.exp(decision))
        proba_real = 1 - proba_fake
    
    return {
        "prediction": int(prediction),
        "label": "Real News ✓" if prediction == 1 else "Fake News ✗",
        "confidence": confidence,
        "proba_real": proba_real,
        "proba_fake": proba_fake,
    }


def extract_feature_details(text: str, extractor) -> dict:
    """Extract detailed feature information for a text."""
    # Get linguistic features (extract takes a single string, returns dict)
    ling_features = extractor._linguistic_extractor.extract(text)
    
    # Get sentiment features (extract takes a single string, returns dict)
    sent_features = extractor._sentiment_extractor.extract(text)
    
    return {
        "linguistic": ling_features,
        "sentiment": sent_features,
    }


# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home - Classify Text", "📊 Dataset Statistics", "📈 Model Performance", "ℹ️ About"]
)

# Load models
model, extractor = load_models()

if page == "🏠 Home - Classify Text":
    st.title("🔍 Arabic Fake News Detector")
    st.markdown("Classify Arabic news articles as **Real** or **Fake** using machine learning.")
    
    if model is None or extractor is None:
        st.error("⚠️ Model not found! Please run `python main.py` first to train the model.")
        st.info("Run the following command in your terminal:")
        st.code("python main.py", language="bash")
    else:
        st.success(f"✓ Model loaded: **{model.name}** (F1-Score: {model.val_f1_score:.4f})")
        
        # Input section
        st.subheader("📝 Enter Arabic Text")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Text input with RTL support
            input_text = st.text_area(
                "Paste your Arabic news article here:",
                height=200,
                placeholder="أدخل نص الخبر العربي هنا...",
                help="Enter the title and/or content of the Arabic news article"
            )
        
        with col2:
            st.markdown("**Quick Examples:**")
            if st.button("📰 Sample Real News"):
                st.session_state.sample_text = "الرئيس يعقد اجتماعا مع مجلس الوزراء لمناقشة الميزانية العامة للدولة"
            if st.button("📰 Sample Fake News"):
                st.session_state.sample_text = "عاجل: اكتشاف كنز ضخم تحت الأرض يغير مستقبل البلاد للأبد"
        
        # Use sample text if selected
        if "sample_text" in st.session_state:
            input_text = st.session_state.sample_text
            del st.session_state.sample_text
            st.rerun()
        
        # Classify button
        if st.button("🔍 Classify", type="primary", use_container_width=True):
            if input_text.strip():
                with st.spinner("Analyzing text..."):
                    result = predict_text(input_text, model, extractor)
                    features = extract_feature_details(input_text, extractor)
                
                # Display result
                st.markdown("---")
                st.subheader("📋 Classification Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result["prediction"] == 1:
                        st.markdown(f"""
                        <div class="result-real">
                            <h2>✅ {result['label']}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-fake">
                            <h2>❌ {result['label']}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Real News Probability", f"{result['proba_real']:.1%}")
                
                with col3:
                    st.metric("Fake News Probability", f"{result['proba_fake']:.1%}")
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['proba_real'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Real News Confidence"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#28a745" if result['prediction'] == 1 else "#dc3545"},
                        'steps': [
                            {'range': [0, 30], 'color': "#f8d7da"},
                            {'range': [30, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#d4edda"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature details
                with st.expander("🔬 Feature Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Linguistic Features:**")
                        ling_df = pd.DataFrame([features["linguistic"]]).T
                        ling_df.columns = ["Value"]
                        st.dataframe(ling_df, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Sentiment Features:**")
                        sent_df = pd.DataFrame([features["sentiment"]]).T
                        sent_df.columns = ["Value"]
                        st.dataframe(sent_df, use_container_width=True)
            else:
                st.warning("⚠️ Please enter some text to classify.")


elif page == "📊 Dataset Statistics":
    st.title("📊 Dataset Statistics")
    
    stats = load_dataset_stats()
    
    if stats:
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        total_samples = sum(s["total"] for s in stats.values())
        total_real = sum(s["real"] for s in stats.values())
        total_fake = sum(s["fake"] for s in stats.values())
        
        with col1:
            st.metric("Total Samples", f"{total_samples:,}")
        with col2:
            st.metric("Real News", f"{total_real:,}", delta=f"{total_real/total_samples:.1%}")
        with col3:
            st.metric("Fake News", f"{total_fake:,}", delta=f"{total_fake/total_samples:.1%}")
        
        st.markdown("---")
        
        # Split distribution
        st.subheader("📁 Dataset Splits")
        
        split_data = []
        for split, data in stats.items():
            split_data.append({
                "Split": split.capitalize(),
                "Total": data["total"],
                "Real": data["real"],
                "Fake": data["fake"],
                "Real %": f"{data['real']/data['total']:.1%}",
                "Fake %": f"{data['fake']/data['total']:.1%}",
            })
        
        st.dataframe(pd.DataFrame(split_data), use_container_width=True, hide_index=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of splits
            fig = px.bar(
                pd.DataFrame(split_data),
                x="Split",
                y="Total",
                color="Split",
                title="Samples per Split",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart of class distribution
            fig = px.pie(
                values=[total_real, total_fake],
                names=["Real News", "Fake News"],
                title="Overall Class Distribution",
                color_discrete_sequence=["#28a745", "#dc3545"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Stacked bar for each split
        fig = go.Figure()
        for split, data in stats.items():
            fig.add_trace(go.Bar(name=f"{split.capitalize()} - Real", x=[split.capitalize()], y=[data["real"]], marker_color="#28a745"))
            fig.add_trace(go.Bar(name=f"{split.capitalize()} - Fake", x=[split.capitalize()], y=[data["fake"]], marker_color="#dc3545"))
        
        fig.update_layout(
            barmode='stack',
            title="Class Distribution by Split",
            xaxis_title="Split",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset not found. Please ensure CSV files are in the `dataset/` folder.")

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    
    if model is None:
        st.error("⚠️ Model not found! Please run `python main.py` first to train the model.")
    else:
        st.success(f"✓ Current Model: **{model.name}**")
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Validation F1-Score", f"{model.val_f1_score:.4f}")
        with col2:
            st.metric("Model Type", model.name)
        with col3:
            if model.best_params:
                params_str = ", ".join(f"{k}={v}" for k, v in model.best_params.items())
                st.metric("Best Parameters", params_str[:30] + "..." if len(params_str) > 30 else params_str)
        
        st.markdown("---")
        
        # Classification report
        st.subheader("📋 Classification Report")
        st.text(model.classification_report)
        
        # Feature extractor info
        if extractor:
            st.markdown("---")
            st.subheader("🔧 Feature Extractor Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_tfidf = len(extractor._tfidf_extractor.get_feature_names())
                n_ling = len(extractor._linguistic_extractor.get_feature_names())
                n_sent = len(extractor._sentiment_extractor.get_feature_names())
                
                st.markdown(f"""
                - **TF-IDF Features:** {n_tfidf:,}
                - **Linguistic Features:** {n_ling}
                - **Sentiment Features:** {n_sent}
                - **Total Features:** {n_tfidf + n_ling + n_sent:,}
                """)
            
            with col2:
                # Top TF-IDF features
                st.markdown("**Top 10 TF-IDF Features:**")
                top_features = extractor.get_top_tfidf_features(n=10)
                for i, (name, weight) in enumerate(top_features, 1):
                    st.text(f"{i}. {name} ({weight:.4f})")

elif page == "ℹ️ About":
    st.title("ℹ️ About")
    
    st.markdown("""
    ## Arabic Fake News Detector
    
    This application uses machine learning to classify Arabic news articles as **Real** or **Fake**.
    
    ### 🔧 Technical Details
    
    **Feature Extraction:**
    - TF-IDF vectorization with n-grams (1-2)
    - Linguistic features (word count, sentence count, etc.)
    - Sentiment analysis features
    
    **Models Trained:**
    - Logistic Regression
    - Linear SVM
    - Random Forest
    
    **Dataset:**
    - Arabic news articles
    - Binary classification (Real vs Fake)
    - Pre-split into train/validation/test sets
    
    ### 📚 How to Use
    
    1. **Classify Text:** Enter Arabic text on the home page to get a prediction
    2. **Dataset Statistics:** View information about the training data
    3. **Model Performance:** Check the model's accuracy and metrics
    
    ### 🚀 Getting Started
    
    If the model is not loaded, run the training script first:
    
    ```bash
    python main.py
    ```
    
    Then start the UI:
    
    ```bash
    streamlit run app.py
    ```
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and scikit-learn")
