import pandas as pd
import numpy as np
import re

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

# Function to map label arrays from string to list of names
def parse_label_array(label_str, mapping):
    try:
        label_list = list(map(int, re.findall(r"\d+", label_str)))
        return [mapping[label - 1] if 1 <= label <= len(mapping) else "Unknown" for label in label_list]
    except Exception:
        return []

# Function to extract all score values and return as space-separated string
def parse_score_array_all(score_str):
    try:
        score_list = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", score_str)))
        return " ".join(f"{score:.6f}" for score in score_list)
    except Exception:
        return ""

# Load CSV
df = pd.read_csv("comparison_results_retinanet.csv")

# Apply parsing and mapping
df["ground_truth_labels"] = df["ground_truth_labels"].apply(lambda x: parse_label_array(str(x), LABEL_MAPPING))
df["finetuned_labels"] = df["finetuned_labels"].apply(lambda x: parse_label_array(str(x), LABEL_MAPPING))
df["pretrained_labels"] = df["pretrained_labels"].apply(lambda x: parse_label_array(str(x), COCO_CLASSES))
df["finetuned_scores"] = df["finetuned_scores"].apply(lambda x: parse_score_array_all(str(x)))
df["pretrained_scores"] = df["pretrained_scores"].apply(lambda x: parse_score_array_all(str(x)))

# Save the cleaned dataset
df.to_csv("Cleaned_and_Mapped_Dataset_Retinanet.csv", index=False)
