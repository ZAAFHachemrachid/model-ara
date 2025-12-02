"""Live Training Dashboard for Arabic Fake News Classification.

Interactive UI to visualize training progress and adjust hyperparameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import json
from datetime import datetime
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.feature_extraction import FeatureExtractor


MODELS_DIR = "saved_models"


def ensure_models_dir():
    """Create models directory if it doesn't exist."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)


def save_model_with_metadata(model, model_type: str, metrics: 'TrainingMetrics', params: dict):
    """Save model with timestamp and metadata."""
    ensure_models_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_f1 = max(metrics.f1_scores) if metrics.f1_scores else 0
    best_acc = max(metrics.accuracies) if metrics.accuracies else 0
    
    # Create folder name with key info
    folder_name = f"{timestamp}_{model_type.replace(' ', '_')}_f1_{best_f1:.4f}"
    model_folder = os.path.join(MODELS_DIR, folder_name)
    os.makedirs(model_folder, exist_ok=True)
    
    # Save model
    from src.model_training.models import TrainedModel
    from src.model_training import save_model
    
    trained = TrainedModel(
        name=model_type.replace(" ", ""),
        model=model,
        best_params=params,
        val_f1_score=best_f1,
        classification_report=""
    )
    save_model(trained, os.path.join(model_folder, "model.joblib"))
    
    # Save metadata
    metadata = {
        "model_type": model_type,
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(),
        "params": params,
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "final_f1": metrics.f1_scores[-1] if metrics.f1_scores else 0,
        "final_accuracy": metrics.accuracies[-1] if metrics.accuracies else 0,
    }
    with open(os.path.join(model_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return model_folder


def list_saved_models():
    """List all saved models with their metadata."""
    ensure_models_dir()
    models = []
    
    for folder in sorted(os.listdir(MODELS_DIR), reverse=True):
        folder_path = os.path.join(MODELS_DIR, folder)
        if os.path.isdir(folder_path):
            metadata_path = os.path.join(folder_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata["folder"] = folder
                metadata["path"] = folder_path
                models.append(metadata)
    
    return models


# Page config
st.set_page_config(
    page_title="Training Dashboard",
    page_icon="📈",
    layout="wide",
)


@dataclass
class TrainingMetrics:
    """Store training metrics for visualization."""
    iterations: list = field(default_factory=list)
    f1_scores: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)
    precisions: list = field(default_factory=list)
    recalls: list = field(default_factory=list)
    losses: list = field(default_factory=list)


@st.cache_data
def load_data():
    """Load and prepare datasets."""
    train_df = pd.read_csv("dataset/train.csv")
    val_df = pd.read_csv("dataset/validation.csv")
    test_df = pd.read_csv("dataset/test.csv")
    return train_df, val_df, test_df


@st.cache_resource
def load_feature_extractor():
    """Load or create feature extractor."""
    extractor_path = "feature_extractor.joblib"
    if os.path.exists(extractor_path):
        return FeatureExtractor.load(extractor_path)
    return None


def get_texts(df):
    """Extract combined text from DataFrame."""
    titles = df["title"].fillna("")
    texts = df["text"].fillna("")
    return (titles + " " + texts).tolist()


def train_logistic_regression_live(X_train, y_train, X_val, y_val, max_iter, C, progress_bar, metrics_placeholder, chart_placeholder):
    """Train Logistic Regression with live updates."""
    metrics = TrainingMetrics()
    
    # Train incrementally by increasing max_iter
    step_size = max(1, max_iter // 20)
    
    for current_iter in range(step_size, max_iter + 1, step_size):
        model = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=current_iter,
            solver='lbfgs',
            random_state=42,
            warm_start=False
        )
        
        try:
            model.fit(X_train, y_train)
        except Exception:
            pass
        
        # Evaluate
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted')
        rec = recall_score(y_val, y_pred, average='weighted')
        
        metrics.iterations.append(current_iter)
        metrics.f1_scores.append(f1)
        metrics.accuracies.append(acc)
        metrics.precisions.append(prec)
        metrics.recalls.append(rec)
        
        # Update progress
        progress_bar.progress(current_iter / max_iter)
        
        # Update metrics display
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("F1-Score", f"{f1:.4f}")
            col2.metric("Accuracy", f"{acc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
        
        # Update chart
        with chart_placeholder.container():
            fig = create_metrics_chart(metrics)
            st.plotly_chart(fig, use_container_width=True, key=f"lr_chart_{current_iter}")
        
        time.sleep(0.1)
    
    return model, metrics


def train_svm_live(X_train, y_train, X_val, y_val, max_iter, C, progress_bar, metrics_placeholder, chart_placeholder):
    """Train Linear SVM with live updates."""
    metrics = TrainingMetrics()
    
    step_size = max(1, max_iter // 20)
    
    for current_iter in range(step_size, max_iter + 1, step_size):
        model = LinearSVC(
            C=C,
            class_weight='balanced',
            max_iter=current_iter,
            dual='auto',
            random_state=42
        )
        
        try:
            model.fit(X_train, y_train)
        except Exception:
            pass
        
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted')
        rec = recall_score(y_val, y_pred, average='weighted')
        
        metrics.iterations.append(current_iter)
        metrics.f1_scores.append(f1)
        metrics.accuracies.append(acc)
        metrics.precisions.append(prec)
        metrics.recalls.append(rec)
        
        progress_bar.progress(current_iter / max_iter)
        
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("F1-Score", f"{f1:.4f}")
            col2.metric("Accuracy", f"{acc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
        
        with chart_placeholder.container():
            fig = create_metrics_chart(metrics)
            st.plotly_chart(fig, use_container_width=True, key=f"svm_chart_{current_iter}")
        
        time.sleep(0.1)
    
    return model, metrics


def train_random_forest_live(X_train, y_train, X_val, y_val, n_estimators, progress_bar, metrics_placeholder, chart_placeholder):
    """Train Random Forest with live updates (incremental trees)."""
    metrics = TrainingMetrics()
    
    step_size = max(1, n_estimators // 20)
    
    for current_n in range(step_size, n_estimators + 1, step_size):
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('rf', RandomForestClassifier(
                n_estimators=current_n,
                random_state=42,
                n_jobs=-1,
                warm_start=True
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted')
        rec = recall_score(y_val, y_pred, average='weighted')
        
        metrics.iterations.append(current_n)
        metrics.f1_scores.append(f1)
        metrics.accuracies.append(acc)
        metrics.precisions.append(prec)
        metrics.recalls.append(rec)
        
        progress_bar.progress(current_n / n_estimators)
        
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("F1-Score", f"{f1:.4f}")
            col2.metric("Accuracy", f"{acc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
        
        with chart_placeholder.container():
            fig = create_metrics_chart(metrics, x_label="n_estimators")
            st.plotly_chart(fig, use_container_width=True, key=f"rf_chart_{current_n}")
        
        time.sleep(0.1)
    
    return pipeline, metrics


def create_metrics_chart(metrics: TrainingMetrics, x_label: str = "Iterations"):
    """Create a live metrics chart."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Progress", "All Metrics"))
    
    # F1 Score main chart
    fig.add_trace(
        go.Scatter(x=metrics.iterations, y=metrics.f1_scores, mode='lines+markers', name='F1-Score', line=dict(color='#2ecc71', width=3)),
        row=1, col=1
    )
    
    # All metrics
    fig.add_trace(
        go.Scatter(x=metrics.iterations, y=metrics.f1_scores, mode='lines', name='F1-Score', line=dict(color='#2ecc71')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=metrics.iterations, y=metrics.accuracies, mode='lines', name='Accuracy', line=dict(color='#3498db')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=metrics.iterations, y=metrics.precisions, mode='lines', name='Precision', line=dict(color='#9b59b6')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=metrics.iterations, y=metrics.recalls, mode='lines', name='Recall', line=dict(color='#e74c3c')),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_xaxes(title_text=x_label, row=1, col=2)
    fig.update_yaxes(title_text="F1-Score", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 1.05], row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    return fig


# Main UI
st.title("Live Training Dashboard")
st.markdown("Train models with real-time visualization and adjustable hyperparameters")

# Sidebar controls
st.sidebar.header("Training Configuration")

model_type = st.sidebar.selectbox(
    "Model Type",
    ["Logistic Regression", "Linear SVM", "Random Forest"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")

if model_type in ["Logistic Regression", "Linear SVM"]:
    max_iter = st.sidebar.slider("Max Iterations", 100, 5000, 1000, step=100)
    c_value = st.sidebar.select_slider("C (Regularization)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
else:
    n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100, step=10)

# Load data
train_df, val_df, test_df = load_data()
extractor = load_feature_extractor()

if extractor is None:
    st.error("Feature extractor not found! Run `python main.py` first.")
    st.stop()

# Prepare features
@st.cache_data
def prepare_features(_extractor, _train_df, _val_df):
    train_texts = get_texts(_train_df)
    val_texts = get_texts(_val_df)
    X_train = _extractor.transform(train_texts)
    X_val = _extractor.transform(val_texts)
    y_train = _train_df["validity"].values
    y_val = _val_df["validity"].values
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = prepare_features(extractor, train_df, val_df)

# Display data info
col1, col2, col3 = st.columns(3)
col1.metric("Training Samples", f"{len(y_train):,}")
col2.metric("Validation Samples", f"{len(y_val):,}")
col3.metric("Features", f"{X_train.shape[1]:,}")

st.markdown("---")

# Initialize session state for trained model
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.trained_metrics = None
    st.session_state.trained_model_type = None
    st.session_state.trained_params = None

# Training section
if st.button("Start Training", type="primary", use_container_width=True):
    st.subheader(f"Training {model_type}")
    
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    start_time = time.time()
    
    if model_type == "Logistic Regression":
        model, metrics = train_logistic_regression_live(
            X_train, y_train, X_val, y_val, max_iter, c_value,
            progress_bar, metrics_placeholder, chart_placeholder
        )
        params = {"C": c_value, "max_iter": max_iter}
    elif model_type == "Linear SVM":
        model, metrics = train_svm_live(
            X_train, y_train, X_val, y_val, max_iter, c_value,
            progress_bar, metrics_placeholder, chart_placeholder
        )
        params = {"C": c_value, "max_iter": max_iter}
    else:
        model, metrics = train_random_forest_live(
            X_train, y_train, X_val, y_val, n_estimators,
            progress_bar, metrics_placeholder, chart_placeholder
        )
        params = {"n_estimators": n_estimators}
    
    elapsed = time.time() - start_time
    
    # Store in session state
    st.session_state.trained_model = model
    st.session_state.trained_metrics = metrics
    st.session_state.trained_model_type = model_type
    st.session_state.trained_params = params
    
    st.success(f"Training complete in {elapsed:.1f}s")

# Show results and save button if model is trained
if st.session_state.trained_model is not None:
    metrics = st.session_state.trained_metrics
    
    st.markdown("---")
    st.subheader("Final Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Best Metrics:**")
        best_idx = np.argmax(metrics.f1_scores)
        st.write(f"- Best F1-Score: **{metrics.f1_scores[best_idx]:.4f}**")
        st.write(f"- At iteration: **{metrics.iterations[best_idx]}**")
        st.write(f"- Accuracy: **{metrics.accuracies[best_idx]:.4f}**")
    
    with col2:
        if len(metrics.f1_scores) > 1:
            improvement = metrics.f1_scores[-1] - metrics.f1_scores[0]
            st.markdown("**Convergence:**")
            st.write(f"- Initial F1: {metrics.f1_scores[0]:.4f}")
            st.write(f"- Final F1: {metrics.f1_scores[-1]:.4f}")
            st.write(f"- Improvement: {improvement:+.4f}")
    
    # Save button outside training block
    if st.button("💾 Save Model"):
        saved_path = save_model_with_metadata(
            st.session_state.trained_model,
            st.session_state.trained_model_type,
            st.session_state.trained_metrics,
            st.session_state.trained_params
        )
        st.success(f"Model saved to: `{saved_path}`")
        st.rerun()  # Refresh to show in saved models list

# Saved Models Section
st.markdown("---")
st.subheader("📁 Saved Models")

saved_models = list_saved_models()
if saved_models:
    models_df = pd.DataFrame([
        {
            "Created": m.get("created_at", "")[:19].replace("T", " "),
            "Model": m.get("model_type", ""),
            "F1-Score": f"{m.get('best_f1', 0):.4f}",
            "Accuracy": f"{m.get('best_accuracy', 0):.4f}",
            "Params": str(m.get("params", {})),
            "Folder": m.get("folder", ""),
        }
        for m in saved_models
    ])
    st.dataframe(models_df, use_container_width=True, hide_index=True)
else:
    st.info("No saved models yet. Train and save a model to see it here.")

# Test Dataset Evaluation Section
st.markdown("---")
st.subheader("🧪 Test Dataset Evaluation")

@st.cache_data
def prepare_test_features(_extractor, _test_df):
    """Prepare test features."""
    test_texts = get_texts(_test_df)
    X_test = _extractor.transform(test_texts)
    y_test = _test_df["validity"].values
    return X_test, y_test

def compute_learning_curve(model, X_train, y_train, X_test, y_test, n_points=10):
    """Compute learning curve showing train/test error over training set sizes."""
    from sklearn.base import clone
    from sklearn.metrics import log_loss
    
    train_sizes = np.linspace(0.1, 1.0, n_points)
    train_errors = []
    test_errors = []
    train_f1s = []
    test_f1s = []
    sizes = []
    
    for size in train_sizes:
        n_samples = int(len(y_train) * size)
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        try:
            # Clone and fit model on subset
            if hasattr(model, 'named_steps'):
                # Pipeline (Random Forest with SMOTE)
                model_clone = ImbPipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('rf', RandomForestClassifier(
                        n_estimators=model.named_steps['rf'].n_estimators,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
            else:
                model_clone = clone(model)
            
            model_clone.fit(X_subset, y_subset)
            
            # Predictions
            y_train_pred = model_clone.predict(X_subset)
            y_test_pred = model_clone.predict(X_test)
            
            # Calculate error rate (1 - accuracy) as proxy for loss
            train_error = 1 - accuracy_score(y_subset, y_train_pred)
            test_error = 1 - accuracy_score(y_test, y_test_pred)
            
            train_f1 = f1_score(y_subset, y_train_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            train_errors.append(train_error)
            test_errors.append(test_error)
            train_f1s.append(train_f1)
            test_f1s.append(test_f1)
            sizes.append(n_samples)
        except Exception as e:
            continue
    
    return {
        'sizes': sizes,
        'train_errors': train_errors,
        'test_errors': test_errors,
        'train_f1s': train_f1s,
        'test_f1s': test_f1s
    }

def create_loss_chart(learning_data):
    """Create loss/error chart."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Error Rate (Loss Proxy)", "F1-Score Learning Curve"))
    
    # Error rate chart (loss-like)
    fig.add_trace(
        go.Scatter(x=learning_data['sizes'], y=learning_data['train_errors'], 
                   mode='lines+markers', name='Train Error', line=dict(color='#3498db', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=learning_data['sizes'], y=learning_data['test_errors'], 
                   mode='lines+markers', name='Test Error', line=dict(color='#e74c3c', width=2)),
        row=1, col=1
    )
    
    # F1 Score chart
    fig.add_trace(
        go.Scatter(x=learning_data['sizes'], y=learning_data['train_f1s'], 
                   mode='lines+markers', name='Train F1', line=dict(color='#2ecc71', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=learning_data['sizes'], y=learning_data['test_f1s'], 
                   mode='lines+markers', name='Test F1', line=dict(color='#9b59b6', width=2)),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Training Samples", row=1, col=1)
    fig.update_xaxes(title_text="Training Samples", row=1, col=2)
    fig.update_yaxes(title_text="Error Rate", row=1, col=1)
    fig.update_yaxes(title_text="F1-Score", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    return fig

X_test, y_test = prepare_test_features(extractor, test_df)

col_test1, col_test2 = st.columns([2, 1])
with col_test1:
    st.markdown(f"**Test Dataset:** {len(y_test):,} samples")
with col_test2:
    test_class_dist = pd.Series(y_test).value_counts()
    st.markdown(f"Real: {test_class_dist.get(1, 0)} | Fake: {test_class_dist.get(0, 0)}")

if st.session_state.trained_model is not None:
    if st.button("🔬 Evaluate on Test Dataset", type="secondary", use_container_width=True):
        with st.spinner("Evaluating model on test dataset..."):
            model = st.session_state.trained_model
            
            # Direct evaluation
            y_test_pred = model.predict(X_test)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            test_acc = accuracy_score(y_test, y_test_pred)
            test_prec = precision_score(y_test, y_test_pred, average='weighted')
            test_rec = recall_score(y_test, y_test_pred, average='weighted')
            
            st.markdown("### Test Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test F1-Score", f"{test_f1:.4f}")
            col2.metric("Test Accuracy", f"{test_acc:.4f}")
            col3.metric("Test Precision", f"{test_prec:.4f}")
            col4.metric("Test Recall", f"{test_rec:.4f}")
            
            # Compute and show learning curve
            st.markdown("### Learning Curve (Loss Graph)")
            with st.spinner("Computing learning curve..."):
                learning_data = compute_learning_curve(model, X_train, y_train, X_test, y_test)
                fig = create_loss_chart(learning_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_test_pred)
            
            st.markdown("### Confusion Matrix")
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Fake (0)', 'Real (1)'],
                y=['Fake (0)', 'Real (1)'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Gap analysis
            val_f1 = max(st.session_state.trained_metrics.f1_scores)
            gap = val_f1 - test_f1
            
            if gap > 0.05:
                st.warning(f"⚠️ Validation-Test gap: {gap:.4f} - Model may be overfitting")
            elif gap < -0.02:
                st.info(f"ℹ️ Test F1 is higher than validation by {-gap:.4f}")
            else:
                st.success(f"✅ Good generalization! Gap: {gap:.4f}")
else:
    st.info("Train a model first to evaluate it on the test dataset")

# History comparison
st.markdown("---")
st.subheader("🔄 Quick Experiment")
st.markdown("Compare different configurations quickly:")

with st.expander("Run Quick Comparison"):
    if st.button("Compare C values (LR)"):
        results = []
        for c in [0.1, 1.0, 10.0]:
            model = LogisticRegression(C=c, class_weight='balanced', max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            f1 = f1_score(y_val, model.predict(X_val), average='weighted')
            results.append({"C": c, "F1-Score": f1})
        
        df = pd.DataFrame(results)
        st.dataframe(df, hide_index=True)
        fig = px.bar(df, x="C", y="F1-Score", title="F1-Score by C value")
        st.plotly_chart(fig, use_container_width=True)
