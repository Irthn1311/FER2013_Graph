import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
cm_path = r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d16_runs\r\a5b\d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150\confusion_matrix.csv"
metrics_path = r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d16_runs\r\a5b\d16r_a5b_heavy_opt_a4_ce_seed42_accmon_150\test_metrics.csv"
output_path = r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d16_runs\r\a5b\confusion_matrix_plot.png"

# FER2013 class labels (lowercase as in the image)
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Read confusion matrix data
cm_data = pd.read_csv(cm_path)

# Initialize a 7x7 matrix with zeros
cm_matrix = np.zeros((7, 7), dtype=int)

# Fill the matrix
for index, row in cm_data.iterrows():
    t_class = int(row['true_class'])
    p_class = int(row['pred_class'])
    count = int(row['count'])
    cm_matrix[t_class, p_class] = count

# Calculate normalized confusion matrix (row percentages)
cm_normalized = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]

# Create annotations combining counts and percentages
annot_labels = np.empty_like(cm_matrix, dtype=object)
for i in range(cm_matrix.shape[0]):
    for j in range(cm_matrix.shape[1]):
        count = cm_matrix[i, j]
        pct = cm_normalized[i, j] * 100
        annot_labels[i, j] = f"{count}\n{pct:.1f}%"

# Read accuracy from test_metrics.csv
try:
    metrics_df = pd.read_csv(metrics_path)
    accuracy = metrics_df['accuracy'].iloc[0] * 100 # Convert to percentage
except Exception as e:
    print(f"Cannot read accuracy, error: {e}")
    accuracy = 0.0

# Plotting
plt.figure(figsize=(8, 6))

# Heatmap with counts for color intensity, but custom text labels
ax = sns.heatmap(cm_matrix, annot=annot_labels, fmt='', cmap='Blues',
                 xticklabels=classes, yticklabels=classes,
                 annot_kws={"size": 10}, cbar_kws={'pad': 0.05})

# Title and labels to match the provided image
plt.title(f'Confusion matrix on test set, acc: {accuracy:.2f}%', fontsize=13, pad=10)
plt.ylabel('True label', fontsize=11)
plt.xlabel('Pred label', fontsize=11)

# X and Y tick labels formatting
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=90, va='center', fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved Confusion Matrix at: {output_path}")
