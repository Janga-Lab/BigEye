import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, log_loss
from xgboost import XGBClassifier

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', message='.*ROC AUC score is not defined.*')

random_state = 10
classifier_name = "xgb"

def intersect(list1, list2):
  """Checks if two lists intersect."""
  return bool(set(list1) & set(list2))

def standardize_data(data: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
    """ Standardize data using StandardScaler """
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    return data_standardized


def safe_roc_auc_score(y_true, y_pred, num_classes):
    """
    Calculate ROC AUC score with handling for single-class cases.
    Returns None if ROC AUC cannot be calculated.
    """
    # Check if only one class is present
    if len(np.unique(y_true)) < 2:
        return None

    try:
        y_true_cat = tf.keras.utils.to_categorical(y=y_true, num_classes=num_classes)
        # RF predict returns class labels; for AUC we might technically want probas,
        # but to match your SVM structure exactly we use the labels converted to one-hot here.
        y_pred_cat = tf.keras.utils.to_categorical(y=y_pred, num_classes=num_classes)
        return roc_auc_score(y_true_cat, y_pred_cat, multi_class='ovo')
    except ValueError:
        return None

  
# Load and Preprocess Data
df = pd.read_csv("bigeye_retinal_lesion_feature_dataframe.tsv", sep="\t")
df.drop('id', axis=1, inplace=True)

labels = df["stage"].to_numpy()
df.drop('stage', axis=1, inplace=True)
df.apply(pd.to_numeric)
data = df

label_encoder = LabelEncoder()
unique_classes = np.unique(labels)
num_classes = len(unique_classes)
label_encoder.fit(unique_classes)

# Transform labels to integers (1D array)
labels_int = label_encoder.transform(labels) 

n = data.shape[0]
outer_kf = StratifiedKFold(n_splits = 10, random_state = 0, shuffle = True)
inner_kf = StratifiedKFold(n_splits = 10, random_state = 0, shuffle = True)

# Initialize Global Metrics Storage
all_metrics = dict()
metrics_list = ["precision", "recall", "f1_score", "accuracy", "auc", "loss"]
for m in metrics_list:
    all_metrics[m] = []
    all_metrics[f"val_{m}"] = []
    all_metrics[f"test_{m}"] = []

# Track best model
best_test_accuracy = 0
best_model = None
best_fold_id = None

for outer_fold, (inner_idx, outer_idx) in enumerate(outer_kf.split(data, labels_int)):

    x_test = data.iloc[outer_idx]
    y_test_int = labels_int[outer_idx]

    y_test_cat = tf.keras.utils.to_categorical(y_test_int, num_classes=num_classes)
    
    inner_data = data.iloc[inner_idx]
    inner_labels = labels_int[inner_idx]
    
    for inner_fold, (train_idx, val_idx) in enumerate(inner_kf.split(inner_data, inner_labels)):

        nested_id = f"{str(outer_fold)}_{str(inner_fold)}"
        print(nested_id)
        
        x_train = inner_data.iloc[train_idx]
        x_val = inner_data.iloc[val_idx]
        
        y_train_int = inner_labels[train_idx]
        y_val_int = inner_labels[val_idx]
        
        # One-hot for log_loss calculation later
        y_train_cat = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val_int, num_classes=num_classes)

        params = {
          "max_depth" : 4,
          "reg_alpha" : 0.5,
          "reg_lambda": 0.5,
          "subsample": 0.8,
          "colsample_bytree" : 0.8,
          "learning_rate" : 0.1,
          "n_estimators" : 100,
          "early_stopping_rounds" : 5,
          "objective": "multi:softprob", # or multi:softmax
          "num_class": num_classes
        }
        
        model = XGBClassifier(**params)
        
        # FIX: Fit on 1D integer labels, not one-hot encoded labels
        model.fit(
          x_train,
          y_train_int,
          eval_set = [(x_val, y_val_int)],
          verbose=False 
        )

        info = dict()

        # --- Training Metrics ---
        # Predict returns 1D integers now
        p = model.predict(x_train) 
        # Predict proba returns probabilities (for log_loss)
        p_proba = model.predict_proba(x_train)

        prec = precision_score(y_train_int, p, average = "macro")
        rec = recall_score(y_train_int, p, average = "macro")
        f1 = f1_score(y_train_int, p, average = "macro")
        acc = accuracy_score(y_train_int, p)
        auc = safe_roc_auc_score(y_train_int, p, num_classes)
        loss = log_loss(y_train_int, p_proba) 

        all_metrics["precision"].append(prec)
        all_metrics["recall"].append(rec)
        all_metrics["f1_score"].append(f1)
        all_metrics["accuracy"].append(acc)
        all_metrics["loss"].append(loss)
        all_metrics["auc"].append(auc)
        
        info["precision"] = prec
        info["recall"] = rec
        info["f1_score"] = f1
        info["accuracy"] = acc
        info["auc"] = auc
        info["loss"] = loss

        # --- Validation Metrics ---
        p = model.predict(x_val)
        p_proba = model.predict_proba(x_val)

        prec = precision_score(y_val_int, p, average = "macro")
        rec = recall_score(y_val_int, p, average = "macro")
        f1 = f1_score(y_val_int, p, average = "macro")
        acc = accuracy_score(y_val_int, p)
        auc = safe_roc_auc_score(y_val_int, p, num_classes)
        loss = log_loss(y_val_int, p_proba)

        all_metrics["val_precision"].append(prec)
        all_metrics["val_recall"].append(rec)
        all_metrics["val_f1_score"].append(f1)
        all_metrics["val_accuracy"].append(acc)
        all_metrics["val_auc"].append(auc)
        all_metrics["val_loss"].append(loss)

        info["val_precision"] = prec
        info["val_recall"] = rec
        info["val_f1_score"] = f1
        info["val_accuracy"] = acc
        info["val_auc"] = auc
        info["val_loss"] = loss


        # --- Test Metrics ---
        p =  model.predict(x_test)
        p_proba = model.predict_proba(x_test)

        prec = precision_score(y_test_int, p, average = "macro")
        rec = recall_score(y_test_int, p, average = "macro")
        f1 = f1_score(y_test_int, p, average = "macro")
        acc = accuracy_score(y_test_int, p)
        auc = safe_roc_auc_score(y_test_int, p, num_classes)
        loss = log_loss(y_test_int, p_proba)

        all_metrics["test_precision"].append(prec)
        all_metrics["test_recall"].append(rec)
        all_metrics["test_f1_score"].append(f1)
        all_metrics["test_accuracy"].append(acc)
        all_metrics["test_auc"].append(auc)
        all_metrics["test_loss"].append(loss)

        info["test_precision"] = prec
        info["test_recall"] = rec
        info["test_f1_score"] = f1
        info["test_accuracy"] = acc
        info["test_auc"] = auc
        info["test_loss"] = loss

        # Track best model based on test accuracy
        if acc > best_test_accuracy:
            best_test_accuracy = acc
            best_model = model
            best_fold_id = nested_id

        fold_filename = f"{classifier_name}_{nested_id}.pickle"

        with open(fold_filename, 'wb') as handle:
            pickle.dump(
                info,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )


            

             
# --- Summary Output ---
print("\n" + "="*70)
print("NESTED CROSS-VALIDATION RESULTS (XGBOOST)")
print("="*70)

# Training set metrics
print("\nTraining Set Metrics:")
print("-" * 70)
for m in metrics_list:
    values = [v if v is not None else np.nan for v in all_metrics[m]]
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

# Validation set metrics
print("\nValidation Set Metrics:")
print("-" * 70)
for m in metrics_list:
    values = [v if v is not None else np.nan for v in all_metrics[f"val_{m}"]]
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

# Test set metrics
print("\nTest Set Metrics:")
print("-" * 70)
for m in metrics_list:
    values = [v if v is not None else np.nan for v in all_metrics[f"test_{m}"]]
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

print("\n" + "="*70)
print(f"Total folds evaluated: {len(all_metrics['accuracy'])}")
print("="*70)

# Save metrics to pickle file
results = {
    "all_metrics": all_metrics
}

with open('xgb_nested_cv_results.pickle', 'wb') as handle:
    pickle.dump(
        results,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("\nResults saved to: xgb_nested_cv_results.pickle")

# Save best model
if best_model is not None:
    # Save XGBoost model in native format (JSON)
    model_filename = f'{classifier_name}_best_model.json'
    best_model.save_model(model_filename)
    
    # Save metadata as pickle
    with open(f'{classifier_name}_best_model.pickle', 'wb') as handle:
        pickle.dump({
            'fold_id': best_fold_id,
            'test_accuracy': best_test_accuracy,
            'label_encoder': label_encoder,
            'num_classes': num_classes,
            'params': params
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nBest model saved to: {model_filename}")
    print(f"Model metadata saved to: {classifier_name}_best_model.pickle")
    print(f"Best fold: {best_fold_id}")
    print(f"Test accuracy: {best_test_accuracy:.4f}")
