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
        finetuned_labels_list = ast.literal_eval(finetuned_labels)  # Convert string to list
        finetuned_scores_list = finetuned_scores  # Already processed as list
        pretrained_labels_list = ast.literal_eval(pretrained_labels)
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
    
    return df[["image_id", "ground_truth_labels", "filtered_finetuned_labels", "filtered_finetuned_scores", "filtered_pretrained_labels", "filtered_pretrained_scores"]]  # Keep only selected columns

# Function to calculate precision, recall, F1-score, and accuracy
def compute_metrics(pred_labels, true_labels):
    tp = sum(1 for label in pred_labels if label in true_labels)
    fp = sum(1 for label in pred_labels if label not in true_labels)
    fn = sum(1 for label in true_labels if label not in pred_labels)
    tn = 0  # Not applicable for multi-label classification
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return precision, recall, f1, accuracy

# Load the dataset
@st.cache_data
def load_data():
    file_path = "Cleaned_and_Mapped_Dataset.csv"  # Load the new cleaned dataset
    df = pd.read_csv(file_path)
    
    # Convert scores from strings to lists of floats
    df["finetuned_scores"] = df["finetuned_scores"].apply(lambda x: process_scores(str(x)))
    df["pretrained_scores"] = df["pretrained_scores"].apply(lambda x: process_scores(str(x)))
    
    # Save processed dataframe
    df.to_csv("processed_cleaned_dataset.csv", index=False)
    
    return df

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
st.dataframe(df_filtered)

# Compute performance metrics
st.subheader("Performance Metrics Comparison")
finetuned_metrics = [compute_metrics(pred, true) for pred, true in zip(df_filtered["filtered_finetuned_labels"], df_filtered["ground_truth_labels"])]
pretrained_metrics = [compute_metrics(pred, true) for pred, true in zip(df_filtered["filtered_pretrained_labels"], df_filtered["ground_truth_labels"])]

finetuned_precision = np.mean([m[0] for m in finetuned_metrics])
finetuned_recall = np.mean([m[1] for m in finetuned_metrics])
finetuned_f1 = np.mean([m[2] for m in finetuned_metrics])
finetuned_accuracy = np.mean([m[3] for m in finetuned_metrics])

pretrained_precision = np.mean([m[0] for m in pretrained_metrics])
pretrained_recall = np.mean([m[1] for m in pretrained_metrics])
pretrained_f1 = np.mean([m[2] for m in pretrained_metrics])
pretrained_accuracy = np.mean([m[3] for m in pretrained_metrics])

st.write("### Fine-tuned Model Metrics")

st.write(f"**Accuracy:** {finetuned_accuracy:.2%}")
st.write(f"**Precision:** {finetuned_precision:.2%}")
st.write(f"**Recall:** {finetuned_recall:.2%}")
st.write(f"**F1-Score:** {finetuned_f1:.2%}")

st.write("### Pre-trained Model Metrics")
st.write(f"**Accuracy:** {pretrained_accuracy:.2%}")
st.write(f"**Precision:** {pretrained_precision:.2%}")
st.write(f"**Recall:** {pretrained_recall:.2%}")
st.write(f"**F1-Score:** {pretrained_f1:.2%}")
