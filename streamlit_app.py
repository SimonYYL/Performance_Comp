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
        
        finetuned_labels_list = ast.literal_eval(finetuned_labels) if isinstance(finetuned_labels, str) else finetuned_labels
        pretrained_labels_list = ast.literal_eval(pretrained_labels) if isinstance(pretrained_labels, str) else pretrained_labels
        
        filtered_finetuned = [(lbl, scr) for lbl, scr in zip(finetuned_labels_list, finetuned_scores) if scr >= threshold]
        filtered_pretrained = [(lbl, scr) for lbl, scr in zip(pretrained_labels_list, pretrained_scores) if scr >= threshold]
        
        filtered_finetuned_labels.append([x[0] for x in filtered_finetuned])
        filtered_finetuned_scores.append([x[1] for x in filtered_finetuned])
        filtered_pretrained_labels.append([x[0] for x in filtered_pretrained])
        filtered_pretrained_scores.append([x[1] for x in filtered_pretrained])
    
    df["filtered_finetuned_labels"] = filtered_finetuned_labels
    df["filtered_finetuned_scores"] = filtered_finetuned_scores
    df["filtered_pretrained_labels"] = filtered_pretrained_labels
    df["filtered_pretrained_scores"] = filtered_pretrained_scores
    
    return df[["image_id", "ground_truth_labels", "filtered_finetuned_labels", "filtered_finetuned_scores", "filtered_pretrained_labels", "filtered_pretrained_scores"]]

# Function to compute metrics
def compute_metrics(pred_labels_list, true_labels_list, use_global=False, label_filter="All"):
    def filter_labels(labels):
        if label_filter == "Person":
            return [lbl for lbl in labels if lbl.lower() == "person"]
        elif label_filter == "Object (Not person)":
            return [lbl for lbl in labels if lbl.lower() != "person"]
        return labels

    if use_global:
        total_tp = total_fp = total_fn = 0

        for pred_labels, true_labels in zip(pred_labels_list, true_labels_list):
            pred_set = set(filter_labels(pred_labels))
            true_set = set(filter_labels(true_labels))

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision > 0 or recall > 0) else 0

        return precision, recall, f1
    else:
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for pred_labels, true_labels in zip(pred_labels_list, true_labels_list):
            pred_set = set(filter_labels(pred_labels))
            true_set = set(filter_labels(true_labels))

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
                f1 = (2 * precision * recall) / (precision + recall)
            else:
                f1 = np.nan


            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        valid_mask = [
            not np.isnan(p) and not np.isnan(r) and not np.isnan(f)
            for p, r, f in zip(precision_scores, recall_scores, f1_scores)]

        filtered_p = [p for p, m in zip(precision_scores, valid_mask) if m]
        filtered_r = [r for r, m in zip(recall_scores, valid_mask) if m]
        filtered_f = [f for f, m in zip(f1_scores, valid_mask) if m]

        avg_precision = np.mean(filtered_p)
        avg_recall = np.mean(filtered_r)
        avg_f1 = np.mean(filtered_f)



        return avg_precision, avg_recall, avg_f1

# --- Title and Sidebar ---
st.title("Model Performance Comparison")

st.sidebar.title("Model Options")
available_models = {
    "Fine-tuned Fast R-CNN": "Cleaned_and_Mapped_Dataset.csv",
    "Fine-tuned Retinanet": "Cleaned_and_Mapped_Dataset_Retinanet.csv",
}
selected_model = st.sidebar.selectbox("Select a model to evaluate", list(available_models.keys()))
selected_file = available_models[selected_model]

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["finetuned_scores"] = df["finetuned_scores"].apply(lambda x: process_scores(str(x)))
    df["pretrained_scores"] = df["pretrained_scores"].apply(lambda x: process_scores(str(x)))
    df["ground_truth_labels"] = df["ground_truth_labels"].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
    return df

df = load_data(selected_file)

# --- Dataset Preview ---
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --- Confidence Threshold + Toggles ---
st.subheader("Filter by Confidence Score")
threshold = st.slider("Select Confidence Score Threshold", 0.0, 1.0, 0.5, 0.05)

use_global_metrics = st.checkbox("Use cumulative (global) metrics instead of per-row averaging", value=False)

label_filter = st.selectbox(
    "Select label category to evaluate:",
    ["All", "Person", "Object (Not person)"]
)

# --- Filter Predictions ---
df_filtered = filter_predictions(df, threshold)
st.dataframe(df_filtered, height=200, width=800)

# --- Metrics Comparison ---
st.subheader("Performance Metrics Comparison")

# Color choices based on selection
mode_color = "#004AAD" if use_global_metrics else "#A6C8FF"  # deep blue vs light blue
label_color_map = {
    "All": "#A6C8FF",                  # neutral gray
    "Person": "#E74C3C",               # red for person
    "Object (Not person)": "#27AE60"   # green for object
}
label_color = label_color_map.get(label_filter, "#666666")

# Styled markdown block
st.markdown(f"""
<div style='line-height: 1.6'>
    <span style='font-weight:bold;'>Metric Calculation Mode:</span>
    <span style='color:{mode_color}; font-weight:bold;'>{'Global (Cumulative)' if use_global_metrics else 'Per-row Averaged'}</span><br>
    <span style='font-weight:bold;'>Label Category:</span>
    <span style='color:{label_color}; font-weight:bold;'>{label_filter}</span>
</div>
""", unsafe_allow_html=True)


finetuned_precision, finetuned_recall, finetuned_f1 = compute_metrics(
    df_filtered["filtered_finetuned_labels"],
    df_filtered["ground_truth_labels"],
    use_global_metrics,
    label_filter
)

pretrained_precision, pretrained_recall, pretrained_f1 = compute_metrics(
    df_filtered["filtered_pretrained_labels"],
    df_filtered["ground_truth_labels"],
    use_global_metrics,
    label_filter
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Fine-tuned Model Metrics")
    st.markdown(f"**Precision:** {finetuned_precision:.2%}")
    st.markdown(f"**Recall:** {finetuned_recall:.2%}")
    st.markdown(f"**F1-Score:** {finetuned_f1:.2%}")

with col2:
    st.markdown("### Pre-trained Model Metrics")
    st.markdown(f"**Precision:** {pretrained_precision:.2%}")
    st.markdown(f"**Recall:** {pretrained_recall:.2%}")
    st.markdown(f"**F1-Score:** {pretrained_f1:.2%}")


# --- Metric Explanation ---
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


import altair as alt

# --- Line Chart: Metric vs Threshold ---
st.subheader("Metrics vs Confidence Threshold")

# Select models to visualize
selected_models_for_plot = st.multiselect(
    "Select models to compare",
    options=list(available_models.keys()),
    default=[selected_model]  # default to current selection
)

# Choose metrics to show
metrics_to_plot = st.multiselect(
    "Select metrics to plot",
    options=["Precision", "Recall", "F1-Score"],
    default=["F1-Score"]
)

# Step size for thresholds
threshold_steps = np.arange(0.0, 1.01, 0.05)

plot_data = []

for model_name in selected_models_for_plot:
    model_file = available_models[model_name]
    model_df = load_data(model_file)

    for threshold in threshold_steps:
        df_filtered_loop = filter_predictions(model_df.copy(), threshold)

        finetuned_metrics = compute_metrics(
            df_filtered_loop["filtered_finetuned_labels"],
            df_filtered_loop["ground_truth_labels"],
            use_global_metrics,
            label_filter
        )

        pretrained_metrics = compute_metrics(
            df_filtered_loop["filtered_pretrained_labels"],
            df_filtered_loop["ground_truth_labels"],
            use_global_metrics,
            label_filter
        )

        for model_type, metrics in zip(["Fine-tuned", "Pre-trained"], [finetuned_metrics, pretrained_metrics]):
            for metric_name, metric_value in zip(["Precision", "Recall", "F1-Score"], metrics):
                if metric_name in metrics_to_plot:
                    plot_data.append({
                        "Threshold": threshold,
                        "Metric": metric_name,
                        "Value": metric_value,
                        "Model": model_name,
                        "Model Type": model_type,
                        "Label Type": label_filter,
                        "Metric Mode": "Global" if use_global_metrics else "Per-row"
                    })

# Create DataFrame for plotting
plot_df = pd.DataFrame(plot_data)

# Altair Chart
chart = alt.Chart(plot_df).mark_line(point=True).encode(
    x=alt.X('Threshold:Q'),
    y=alt.Y('Value:Q'),
    color=alt.Color('Model Type:N'),
    strokeDash='Model:N',
    tooltip=['Model', 'Model Type', 'Label Type', 'Metric Mode', 'Threshold', 'Metric', 'Value']
).facet(
    column='Metric:N'
).properties(
    title="Metric Change vs Confidence Threshold"
)

st.altair_chart(chart, use_container_width=True)
