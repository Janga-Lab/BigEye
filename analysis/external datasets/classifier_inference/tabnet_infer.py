import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_DIR = 'tabnet_best_model'
MODEL_METADATA_PATH = 'tabnet_best_model.pickle'
TEST_DATA_PATH = 'idrid_metadata_with_lesions.csv'

# Feature columns - MUST match training data order exactly
# Training data column order from bigeye_retinal_lesion_feature_dataframe.tsv:
# microaneurysm count, microaneurysm area, exudate count, exudate area,
# hemorrhage count, hemorrhage area, cotton wool spot count, cotton wool spot area,
# neovascularization count, neovascularization area,
# laser photocoagulation scar count, laser photocoagulation scar area
FEATURE_COLUMNS = [
    'microaneurysm count',
    'microaneurysm area',
    'exudate count',
    'exudate area',
    'hemorrhage count',
    'hemorrhage area',
    'cotton wool spot count',
    'cotton wool spot area',
    'neovascularization count',
    'neovascularization area',
    'laser photocoagulation scar count',
    'laser photocoagulation scar area'
]

# Label column in test data
LABEL_COLUMN = 'dr_stage'

# ============================================================================
# LOAD MODEL
# ============================================================================
print("Loading trained TabNet model...")
model = load_model(MODEL_DIR)

# Load metadata
with open(MODEL_METADATA_PATH, 'rb') as handle:
    metadata = pickle.load(handle)

label_encoder = metadata['label_encoder']
num_classes = metadata['num_classes']
fold_id = metadata['fold_id']
train_test_accuracy = metadata['test_accuracy']
tabnet_params = metadata['tabnet_params']

print(f"Model loaded from fold: {fold_id}")
print(f"Training test accuracy: {train_test_accuracy:.4f}")
print(f"\nTabNet parameters:")
for key, value in tabnet_params.items():
    print(f"  {key}: {value}")

print(f"\nClass mapping:")
for class_name, class_idx in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"  {class_idx}: {class_name}")

print(f"\nModel architecture:")
model.summary()

# ============================================================================
# LOAD TEST DATA
# ============================================================================
print(f"\nLoading test data from: {TEST_DATA_PATH}")
df_test = pd.read_csv(TEST_DATA_PATH)
print(f"Test dataset size: {len(df_test)} samples")

# Extract features in the correct order
X_test = df_test[FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce')

# Check for missing values
print(f"\nChecking for missing values in features:")
missing_counts = X_test.isnull().sum()
if missing_counts.sum() > 0:
    print("Missing values found:")
    for col in FEATURE_COLUMNS:
        if missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} missing values")
    print("\nFilling missing values with 0...")
    X_test = X_test.fillna(0)
else:
    print("No missing values found.")

# Convert to numpy array
X_test = X_test.values

print(f"\nFeature matrix shape: {X_test.shape}")
print(f"Expected features: {len(FEATURE_COLUMNS)}")

# Extract and encode labels
y_test_labels = df_test[LABEL_COLUMN].to_numpy()

# Check if all test labels are in the training label encoder
unknown_labels = set(y_test_labels) - set(label_encoder.classes_)
if unknown_labels:
    print(f"\nWARNING: Test set contains labels not seen during training: {unknown_labels}")
    print("These samples will be excluded from evaluation.")
    # Filter out unknown labels
    valid_mask = np.isin(y_test_labels, label_encoder.classes_)
    X_test = X_test[valid_mask]
    y_test_labels = y_test_labels[valid_mask]
    df_test = df_test[valid_mask].reset_index(drop=True)
    print(f"Filtered dataset size: {len(df_test)} samples")

y_test_encoded = label_encoder.transform(y_test_labels)

# Convert to one-hot for TabNet
y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

print(f"\nTest set label distribution:")
for label in label_encoder.classes_:
    count = np.sum(y_test_labels == label)
    print(f"  {label}: {count} samples ({100*count/len(y_test_labels):.1f}%)")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================
print("\nMaking predictions...")
# Get probability predictions
y_pred_proba = model.predict(X_test, verbose=0)

# Convert probabilities to class predictions
y_pred = np.argmax(y_pred_proba, axis=1)

# Decode predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels_decoded = label_encoder.inverse_transform(y_test_encoded)

# ============================================================================
# CALCULATE METRICS
# ============================================================================
print("\n" + "="*70)
print("TEST SET PERFORMANCE METRICS")
print("="*70)

# Evaluate model (returns loss and metrics)
test_performance = model.evaluate(X_test, y_test_categorical, batch_size=32, verbose=0)
test_loss = test_performance[0]
test_accuracy_keras = test_performance[1]
test_recall_keras = test_performance[2]
test_precision_keras = test_performance[3]
test_auc_keras = test_performance[4]

print(f"\nKeras Model Evaluation:")
print(f"  Loss:      {test_loss:.4f}")
print(f"  Accuracy:  {test_accuracy_keras:.4f}")
print(f"  Precision: {test_precision_keras:.4f}")
print(f"  Recall:    {test_recall_keras:.4f}")
print(f"  AUC:       {test_auc_keras:.4f}")

# Overall metrics using sklearn
accuracy = accuracy_score(y_test_encoded, y_pred)
precision_macro = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)

print(f"\nOverall Metrics (Macro-averaged - sklearn):")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision_macro:.4f}")
print(f"  Recall:    {recall_macro:.4f}")
print(f"  F1-Score:  {f1_macro:.4f}")

# Weighted metrics (accounts for class imbalance)
precision_weighted = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)

print(f"\nWeighted Metrics (Accounts for class imbalance):")
print(f"  Precision: {precision_weighted:.4f}")
print(f"  Recall:    {recall_weighted:.4f}")
print(f"  F1-Score:  {f1_weighted:.4f}")

# Per-class metrics
print("\n" + "-"*70)
print("Per-Class Classification Report:")
print("-"*70)
print(classification_report(
    y_test_labels_decoded, 
    y_pred_labels, 
    labels=label_encoder.classes_,
    zero_division=0
))

# ============================================================================
# CONFUSION MATRIX
# ============================================================================
cm = confusion_matrix(y_test_labels_decoded, y_pred_labels, labels=label_encoder.classes_)

print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=label_encoder.classes_
)
disp.plot(ax=ax, cmap='RdPu', values_format='d')
plt.title('Confusion Matrix - TabNet Model on External Test Set')
plt.tight_layout()
plt.savefig('tabnet_confusion_matrix_external_test.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved to: tabnet_confusion_matrix_external_test.png")

# ============================================================================
# PREDICTION CONFIDENCE ANALYSIS (TabNet specific)
# ============================================================================
print("\n" + "="*70)
print("PREDICTION CONFIDENCE ANALYSIS")
print("="*70)

# Get max probability (confidence) for each prediction
prediction_confidences = np.max(y_pred_proba, axis=1)

print(f"\nPrediction Confidence Statistics:")
print(f"  Mean confidence:   {np.mean(prediction_confidences):.4f}")
print(f"  Median confidence: {np.median(prediction_confidences):.4f}")
print(f"  Min confidence:    {np.min(prediction_confidences):.4f}")
print(f"  Max confidence:    {np.max(prediction_confidences):.4f}")

# Confidence by correctness
correct_mask = y_pred == y_test_encoded
correct_confidences = prediction_confidences[correct_mask]
incorrect_confidences = prediction_confidences[~correct_mask]

if len(correct_confidences) > 0:
    print(f"\nCorrect predictions:")
    print(f"  Mean confidence: {np.mean(correct_confidences):.4f}")
if len(incorrect_confidences) > 0:
    print(f"\nIncorrect predictions:")
    print(f"  Mean confidence: {np.mean(incorrect_confidences):.4f}")

# Visualize confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of all confidences
axes[0].hist(prediction_confidences, bins=30, alpha=0.7, color='mediumvioletred', edgecolor='black')
axes[0].set_xlabel('Prediction Confidence')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Prediction Confidences')
axes[0].grid(True, alpha=0.3)

# Confidence by correctness
if len(correct_confidences) > 0 and len(incorrect_confidences) > 0:
    axes[1].hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[1].hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[1].set_xlabel('Prediction Confidence')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Confidence Distribution by Correctness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tabnet_confidence_analysis.png', dpi=300, bbox_inches='tight')
print("\nConfidence analysis plot saved to: tabnet_confidence_analysis.png")

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================
# Create results dataframe
results_df = df_test.copy()
results_df['predicted_stage'] = y_pred_labels
results_df['true_stage'] = y_test_labels_decoded
results_df['correct'] = results_df['predicted_stage'] == results_df['true_stage']
results_df['prediction_confidence'] = prediction_confidences

# Add probability columns for each class
for i, class_name in enumerate(label_encoder.classes_):
    results_df[f'prob_{class_name}'] = y_pred_proba[:, i]

# Save to CSV
output_path = 'tabnet_predictions_external_test.csv'
results_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to: {output_path}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total test samples: {len(y_test_encoded)}")
print(f"Correctly classified: {np.sum(y_pred == y_test_encoded)} ({100*accuracy:.2f}%)")
print(f"Misclassified: {np.sum(y_pred != y_test_encoded)} ({100*(1-accuracy):.2f}%)")
print("\nMost common misclassifications:")
misclassified_pairs = []
for i in range(len(y_test_encoded)):
    if y_pred[i] != y_test_encoded[i]:
        true_label = y_test_labels_decoded[i]
        pred_label = y_pred_labels[i]
        misclassified_pairs.append((true_label, pred_label))

if misclassified_pairs:
    from collections import Counter
    pair_counts = Counter(misclassified_pairs)
    for (true_label, pred_label), count in pair_counts.most_common(5):
        print(f"  {true_label} → {pred_label}: {count} times")
else:
    print("  None (perfect classification!)")

# Low confidence predictions
low_confidence_threshold = 0.5
low_confidence_mask = prediction_confidences < low_confidence_threshold
n_low_confidence = np.sum(low_confidence_mask)
if n_low_confidence > 0:
    print(f"\nLow confidence predictions (< {low_confidence_threshold}):")
    print(f"  Count: {n_low_confidence} ({100*n_low_confidence/len(prediction_confidences):.2f}%)")
    low_conf_correct = np.sum(correct_mask[low_confidence_mask])
    print(f"  Correct: {low_conf_correct} ({100*low_conf_correct/n_low_confidence:.2f}%)")

print("="*70)

# ============================================================================
# TABNET-SPECIFIC: FEATURE IMPORTANCE (via attention masks)
# ============================================================================
print("\n" + "="*70)
print("TABNET FEATURE IMPORTANCE (Attention-based)")
print("="*70)
print("\nNote: TabNet uses attention masks to determine feature importance.")
print("Higher values indicate features that received more attention during prediction.")
print("\nTo get detailed feature importance for each sample, TabNet's attention")
print("masks can be extracted during prediction. This requires model modifications")
print("that preserve attention outputs during inference.")
print("="*70)
