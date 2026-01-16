import numpy as np
import pandas as pd
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

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = 'rf_best_model.pickle'
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
print("Loading trained Random Forest model...")
with open(MODEL_PATH, 'rb') as handle:
    model_data = pickle.load(handle)

model = model_data['model']
label_encoder = model_data['label_encoder']
fold_id = model_data['fold_id']
train_test_accuracy = model_data['test_accuracy']

print(f"Model loaded from fold: {fold_id}")
print(f"Training test accuracy: {train_test_accuracy:.4f}")
print(f"\nClass mapping:")
for class_name, class_idx in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"  {class_idx}: {class_name}")

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

# Convert to numpy array (avoids feature name mismatch)
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

print(f"\nTest set label distribution:")
for label in label_encoder.classes_:
    count = np.sum(y_test_labels == label)
    print(f"  {label}: {count} samples ({100*count/len(y_test_labels):.1f}%)")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Decode predictions back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels_decoded = label_encoder.inverse_transform(y_test_encoded)

# ============================================================================
# CALCULATE METRICS
# ============================================================================
print("\n" + "="*70)
print("TEST SET PERFORMANCE METRICS")
print("="*70)

# Overall metrics
accuracy = accuracy_score(y_test_encoded, y_pred)
precision_macro = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)

print(f"\nOverall Metrics (Macro-averaged):")
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
disp.plot(ax=ax, cmap='Greens', values_format='d')
plt.title('Confusion Matrix - Random Forest Model on External Test Set')
plt.tight_layout()
plt.savefig('rf_confusion_matrix_external_test.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved to: rf_confusion_matrix_external_test.png")

# ============================================================================
# FEATURE IMPORTANCE (Random Forest specific)
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:40s}: {row['importance']:.4f}")

# Visualize feature importance
fig, ax = plt.subplots(figsize=(10, 6))
importance_df.plot(x='feature', y='importance', kind='barh', ax=ax, legend=False)
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature')
ax.set_title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved to: rf_feature_importance.png")

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================
# Create results dataframe
results_df = df_test.copy()
results_df['predicted_stage'] = y_pred_labels
results_df['true_stage'] = y_test_labels_decoded
results_df['correct'] = results_df['predicted_stage'] == results_df['true_stage']

# Save to CSV
output_path = 'rf_predictions_external_test.csv'
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

print("="*70)
