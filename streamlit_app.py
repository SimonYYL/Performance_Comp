import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# Label mappings
LABEL_MAPPING = [
    "person", "cell phone", "motorcycle", "handbag", "clock", "book", "cup", "cake", "laptop", "couch", 
    "dog", "fork", "keyboard", "mouse", "chair", "dining table", "bed", "orange", "pizza", "bicycle", "tv", 
    "car", "potted plant", "donut", "wine glass", "knife", "remote", "cat"
]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Function to map labels
def map_labels(label_str, mapping):
    try:
        label_list = ast.literal_eval(label_str)  # Convert string to list safely
        return [mapping[label - 1] if 1 <= label <= len(mapping) else "Unknown" for label in label_list]
    except Exception:
        return []

# Function to process scores
def process_scores(score_str):
    try:
        score_list = [float(x) for x in score_str.replace("[", "").replace("]", "").split()]
        return np.mean(score_list) if score_list else 0.0
    except Exception:
        return 0.0

# Load the dataset
@st.cache_data
def load_data():
    file_path = "comparison_results.csv"  # Replace with the correct file path if needed
    df = pd.read_csv(file_path)
    df["ground_truth_labels"] = df["ground_truth_labels"].apply(lambda x: map_labels(str(x), LABEL_MAPPING))
    df["finetuned_labels"] = df["finetuned_labels"].apply(lambda x: map_labels(str(x), LABEL_MAPPING))
    df["pretrained_labels"] = df["pretrained_labels"].apply(lambda x: map_labels(str(x), COCO_CLASSES))
    df["finetuned_scores"] = df["finetuned_scores"].apply(lambda x: process_scores(str(x)))
    df["pretrained_scores"] = df["pretrained_scores"].apply(lambda x: process_scores(str(x)))
    return df

df = load_data()

# Title
st.title("Model Performance Comparison")

# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Compute performance metrics
def calculate_accuracy(labels, ground_truth):
    correct = 0
    total = len(labels)
    for pred, truth in zip(labels, ground_truth):
        if set(pred) & set(truth):
            correct += 1
    return correct / total if total > 0 else 0

finetuned_accuracy = calculate_accuracy(df["finetuned_labels"], df["ground_truth_labels"])
pretrained_accuracy = calculate_accuracy(df["pretrained_labels"], df["ground_truth_labels"])

# Show metrics
st.subheader("Model Accuracy Comparison")
st.write(f"**Fine-tuned Model Accuracy:** {finetuned_accuracy:.2%}")
st.write(f"**Pre-trained Model Accuracy:** {pretrained_accuracy:.2%}")

# Confidence score visualization
st.subheader("Confidence Score Distribution")
fig, ax = plt.subplots()
ax.boxplot([df["finetuned_scores"], df["pretrained_scores"]],
           labels=["Fine-tuned Model", "Pre-trained Model"])
ax.set_ylabel("Confidence Score")
st.pyplot(fig)

# Scatter plot of scores
st.subheader("Scatter Plot of Confidence Scores")
fig, ax = plt.subplots()
ax.scatter(df.index, df["finetuned_scores"], label="Fine-tuned Model", alpha=0.5)
ax.scatter(df.index, df["pretrained_scores"], label="Pre-trained Model", alpha=0.5)
ax.set_xlabel("Image Index")
ax.set_ylabel("Average Confidence Score")
ax.legend()
st.pyplot(fig)

# Add filters
st.subheader("Filter by Confidence Score")
threshold = st.slider("Select Confidence Score Threshold", 0.0, 1.0, 0.5, 0.05)
filtered_df = df[(df["finetuned_scores"] >= threshold) & (df["pretrained_scores"] >= threshold)]
st.dataframe(filtered_df)