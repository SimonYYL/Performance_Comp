import streamlit as st
import pandas as pd
import numpy as np
import ast

# Function to process scores while keeping all values
def process_scores(score_str):
    try:
        score_list = [float(x) for x in score_str.replace("[", "").replace("]", "").split()]
        return score_list if score_list else []
    except Exception:
        return []

# Function to filter out low-confidence predictions per label
def filter_predictions(df, threshold):
    filtered_finetuned_labels = []
    filtered_finetuned_scores = []
    filtered_pretrained_labels = []
    filtered_pretrained_scores = []
    
    for finetuned_labels, finetuned_scores, pretrained_labels, pretrained_scores in zip(
            df["finetuned_labels"], df["finetuned_scores"], df["pretrained_labels"], df["pretrained_scores"]):
        
        # Convert string lists to actual lists
        finetuned_labels_list = ast.literal_eval(finetuned_labels) if isinstance(finetuned_labels, str) else finetuned_labels
        pretrained_labels_list = ast.literal_eval(pretrained_labels) if isinstance(pretrained_labels, str) else pretrained_labels
        
        finetuned_scores_list = finetuned_scores
        pretrained_scores_list = pretrained_scores

        filtered_finetuned = [(lbl, scr) for lbl, scr in zip(finetuned_labels_list, finetuned_scores_list) if scr >= threshold]
        filtered_pretrained = [(lbl, scr) for lbl, scr in zip(pretrained_labels_list, pretrained_scores_list) if scr >= threshold]
        
        filtered_finetuned_labels.append([x[0] for x in filtered_finetuned])
        filtered_finetuned_scores.append([x[1] for x in filtered_finetuned])
        filtered_pretrained_labels.append([x[0] for x in filtered_pretrained])
        filtered_pretrained_scores.append([x[1] for x in filtered_pretrained])
    
    df["filtered_finetuned_labels"] = filtered_finetuned_labels
    df["filtered_finetuned_scores"] = filtered_finetuned_scores
    df["filtered_pretrained_labels"] = filtered_pretrained_labels
    df["filtered_pretrained_scores"] = filtered_pretrained_scores
    
    return df[["image_id", "ground_truth_labels", "filtered_finetuned_labels", "filtered_finetuned_scores", "filtered_pretrained_labels", "filtered_pretrained_scores"]]

# Function to compute metrics (per-row or global)
def compute_metrics(pred_labels_list, true_labels_list, use_global=False):
    if use_global:
        total_tp = total_fp = total_fn = 0

        for pred_labels, true_labels in zip(pred_labels_list, true_labels_list):
            pred_set = set(pred_labels)
            true_set = set(true_labels)

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1
    else:
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for pred_labels, true_labels in zip(pred_labels_list, true_labels_list):
            pred_set = set(pred_labels)
            true_set = set(true_labels)

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        avg_precision = np.nanmean(precision_scores)
        avg_recall = np.nanmean(recall_scores)
        avg_f1 = np.nanmean(f1_scores)

        return avg_precision, avg_recall, avg_f1

# Title
st.title("Model Performance Comparison")

# --- Sidebar ---
st.sidebar.title("Model Options")

# Model selection
available_models = {
    "Fine-tuned Fast R-CNN": "Cleaned_and_Mapped_Dataset.csv",
    "Fine-tuned Retinanet": "Cleaned_and_Mapped_Dataset_Retinanet.csv",
}
selected_model = st.sidebar.selectbox("Select a model to evaluate", list(available_models.keys()))
selected_file = available_models[selected_model]

# Load the dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["finetuned_scores"] = df["finetuned_scores"].apply(lambda x: process_scores(str(x)))
    df["pretrained_scores"] = df["pretrained_scores"].apply(lambda x: process_scores(str(x)))
    df["ground_truth_labels"] = df["ground_truth_labels"].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
    return df

df = load_data(selected_file)

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Confidence threshold filter
st.subheader("Filter by Confidence Score")
threshold = st.slider("Select Confidence Score Threshold", 0.0, 1.0, 0.5, 0.05)

# Checkbox in the main view, under the slider
use_global_metrics = st.checkbox("Use cumulative (global) metrics instead of per-row averaging", value=False)

# Filter predictions
df_filtered = filter_predictions(df, threshold)
st.dataframe(df_filtered, height=200, width=800)

# Metrics comparison
st.subheader("Performance Metrics Comparison")

# Label the current metric mode
mode_label = "Global (Cumulative)" if use_global_metrics else "Per-row Averaged"
mode_color = "#ff4b4b" if use_global_metrics else "#4BB543"

st.markdown(f"""
<span style="font-weight:bold;">Metric Calculation Mode:</span>
<span style="color:{mode_color}; font-weight:bold;">{mode_label}</span>
""", unsafe_allow_html=True)

# Fine-tuned model metrics
finetuned_precision, finetuned_recall, finetuned_f1 = compute_metrics(
    df_filtered["filtered_finetuned_labels"], df_filtered["ground_truth_labels"], use_global_metrics
)

# Pre-trained model metrics
pretrained_precision, pretrained_recall, pretrained_f1 = compute_metrics(
    df_filtered["filtered_pretrained_labels"], df_filtered["ground_truth_labels"], use_global_metrics
)

st.write("### Fine-tuned Model Metrics")
st.write(f"**Precision:** {finetuned_precision:.2%}")
st.write(f"**Recall:** {finetuned_recall:.2%}")
st.write(f"**F1-Score:** {finetuned_f1:.2%}")

st.write("### Pre-trained Model Metrics")
st.write(f"**Precision:** {pretrained_precision:.2%}")
st.write(f"**Recall:** {pretrained_recall:.2%}")
st.write(f"**F1-Score:** {pretrained_f1:.2%}")

# Metric formula explanation
st.subheader("Evaluation Metrics")

st.markdown("""
#### How the performance is calculated:
For each sample (row), or across the entire dataset (depending on your selection), we compute:

- **True Positives (TP):** Labels correctly predicted and exist in ground truth.
- **False Positives (FP):** Labels predicted but not in ground truth.
- **False Negatives (FN):** Ground truth labels not predicted.

Formulas used:
""")

st.latex(r"""
Precision = \frac{TP}{TP + FP}
""")
st.latex(r"""
Recall = \frac{TP}{TP + FN}
""")
st.latex(r"""
F1\text{-}Score = \frac{2 \times Precision \times Recall}{Precision + Recall}
""")
