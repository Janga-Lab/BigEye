import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, f1_score, log_loss
)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm

# Configuration
DF = "retinal_lesion_features.tsv"
RANDOM_STATE = 10
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 10



# Load and prepare data
df = pd.read_csv(DF, sep="\t")
df.drop(['id', 'image_path', 'prediction_path', 'stage'], axis=1, inplace=True)

# Prepare labels
labels = pd.read_csv(DF, sep="\t")["stage"].to_numpy()
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# LightGBM parameters
lgbm_params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 16,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'early_stopping_rounds': 5,
    'verbose': -1
}

def calculate_metrics(y_true, y_pred, num_classes, prefix=''):
    """Calculate classification metrics with proper error handling."""
    metrics = {
        f'{prefix}precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}accuracy': accuracy_score(y_true, y_pred)
    }
    
    # Calculate AUC and loss with one-hot encoding
    try:
        y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes)
        y_pred_oh = tf.keras.utils.to_categorical(y_pred, num_classes)
        metrics[f'{prefix}auc'] = roc_auc_score(y_true_oh, y_pred_oh, multi_class='ovo')
        metrics[f'{prefix}loss'] = log_loss(y_true_oh, y_pred_oh)
    except (ValueError, Exception):
        metrics[f'{prefix}auc'] = np.nan
        metrics[f'{prefix}loss'] = np.nan
    
    return metrics

# Store all results
all_results = []

# Nested cross-validation
outer_kf = StratifiedKFold(n_splits=N_OUTER_FOLDS, random_state=RANDOM_STATE, shuffle=True)
inner_kf = StratifiedKFold(n_splits=N_INNER_FOLDS, random_state=RANDOM_STATE, shuffle=True)

for outer_fold, (inner_idx, outer_idx) in enumerate(outer_kf.split(df, labels_encoded)):
    x_test = df.iloc[outer_idx]
    y_test = labels_encoded[outer_idx]
    
    inner_data = df.iloc[inner_idx]
    inner_labels = labels_encoded[inner_idx]
    
    for inner_fold, (train_idx, val_idx) in enumerate(inner_kf.split(inner_data, inner_labels)):
        fold_id = f"{outer_fold}_{inner_fold}"
        print(f"Processing fold {fold_id}")
        
        # Split data
        x_train = inner_data.iloc[train_idx]
        x_val = inner_data.iloc[val_idx]
        y_train = inner_labels[train_idx]
        y_val = inner_labels[val_idx]
        
        # Train model
        model = lgbm.LGBMClassifier(**lgbm_params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        
        # Collect metrics for all splits
        fold_results = {'fold_id': fold_id, 'outer_fold': outer_fold, 'inner_fold': inner_fold}
        
        # Training metrics
        y_train_pred = model.predict(x_train)
        fold_results.update(calculate_metrics(y_train, y_train_pred, num_classes, prefix='train_'))
        
        # Validation metrics
        y_val_pred = model.predict(x_val)
        fold_results.update(calculate_metrics(y_val, y_val_pred, num_classes, prefix='val_'))
        
        # Test metrics
        y_test_pred = model.predict(x_test)
        fold_results.update(calculate_metrics(y_test, y_test_pred, num_classes, prefix='test_'))
        
        all_results.append(fold_results)

# Convert to DataFrame and save
results_df = pd.DataFrame(all_results)
results_df.to_csv(f"lgbm_results.tsv", sep='\t', index=False)

# Calculate and save average metrics across all folds
metric_columns = [col for col in results_df.columns if col not in ['fold_id', 'outer_fold', 'inner_fold']]
average_metrics = results_df[metric_columns].mean()
std_metrics = results_df[metric_columns].std()

summary_df = pd.DataFrame({
    'metric': average_metrics.index,
    'mean': average_metrics.values,
    'std': std_metrics.values
})
summary_df.to_csv(f"lgbm_summary.tsv", sep='\t', index=False)

print("\n" + "="*60)
print("Average Metrics Across All Folds:")
print("="*60)
print(summary_df.to_string(index=False))
print("\nResults saved to:")
print(f"  - lgbm_results.tsv (all fold results)")
print(f"  - lgbm_summary.tsv (mean ± std)")
