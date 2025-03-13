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
        
        finetuned_scores_list = finetuned_scores  # Already processed as list
        pretrained_scores_list = pretrained_scores

        # Filter out labels with scores below the threshold
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

# Function to compute multi-label precision, recall, and F1-score
def compute_metrics(pred_labels_list, true_labels_list):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred_labels, true_labels in zip(pred_labels_list, true_labels_list):
        pred_labels_set = set(pred_labels)  # Convert list to set
        true_labels_set = set(true_labels)

        tp = len(pred_labels_set & true_labels_set)  # Intersection
        fp = len(pred_labels_set - true_labels_set)  # Predicted but not in ground truth
        fn = len(true_labels_set - pred_labels_set)  # Ground truth not predicted

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    return avg_precision, avg_recall, avg_f1

# Load the dataset
@st.cache_data
def load_data():
    file_path = "Cleaned_and_Mapped_Dataset.csv"  # Load the new cleaned dataset
    df = pd.read_csv(file_path)
    
    # Convert scores from strings to lists of floats
    df["finetuned_scores"] = df["finetuned_scores"].apply(lambda x: process_scores(str(x)))
    df["pretrained_scores"] = df["pretrained_scores"].apply(lambda x: process_scores(str(x)))
    
    # Ensure ground truth labels are stored as lists
    df["ground_truth_labels"] = df["ground_truth_labels"].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)

    # Save processed dataframe
    df.to_csv("processed_cleaned_dataset.csv", index=False)
    
    return df

# Load dataset
df = load_data()

# Title
st.title("Model Performance Comparison")

# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Add filters
st.subheader("Filter by Confidence Score")
threshold = st.slider("Select Confidence Score Threshold", 0.0, 1.0, 0.5, 0.05)
df_filtered = filter_predictions(df, threshold)
st.dataframe(df_filtered, height=200, width=800)

# Compute performance metrics
st.subheader("Performance Metrics Comparison")

# Compute metrics for fine-tuned model
finetuned_precision, finetuned_recall, finetuned_f1 = compute_metrics(
    df_filtered["filtered_finetuned_labels"], df_filtered["ground_truth_labels"]
)

# Compute metrics for pre-trained model
pretrained_precision, pretrained_recall, pretrained_f1 = compute_metrics(
    df_filtered["filtered_pretrained_labels"], df_filtered["ground_truth_labels"]
)

st.write("### Fine-tuned Model Metrics")
st.write(f"**Precision:** {finetuned_precision:.2%}")
st.write(f"**Recall:** {finetuned_recall:.2%}")
st.write(f"**F1-Score:** {finetuned_f1:.2%}")

st.write("### Pre-trained Model Metrics")
st.write(f"**Precision:** {pretrained_precision:.2%}")
st.write(f"**Recall:** {pretrained_recall:.2%}")
st.write(f"**F1-Score:** {pretrained_f1:.2%}")


st.subheader("Evaluation Metrics")

st.markdown("""
#### How the performance been calculated
For each sample (row), we compute:
- **True Positives (TP):** Labels correctly predicted that exist in ground truth.
- **False Positives (FP):** Labels predicted but not in ground truth.
- **False Negatives (FN):** Ground truth labels that were not predicted.

The formulas used are:
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
