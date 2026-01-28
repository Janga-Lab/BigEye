import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from sklearn.model_selection import KFold, StratifiedKFold

# Suppress UndefinedMetricWarning for ROC AUC when only one class is present
warnings.filterwarnings('ignore', message='.*ROC AUC score is not defined.*')

random_state = 10
classifier_name = "svm"

def intersect(list1, list2):
    """Checks if two lists intersect.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        True if the lists intersect, False otherwise.
    """
    return bool(set(list1) & set(list2))

def standardize_data(data: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
    """ Standardize data using StandardScaler """
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    return data_standardized

def safe_roc_auc_score(y_true, decision_scores, num_classes):
    """
    Calculate ROC AUC score using SVM decision function scores.
    
    FIXED: Now correctly uses decision function scores instead of discrete predictions.
    
    Args:
        y_true: True labels (integer encoded)
        decision_scores: Decision function scores (n_samples x n_classes for OvR)
        num_classes: Number of classes
    
    Returns:
        AUC score or None if calculation fails
    """
    # Check if only one class is present
    if len(np.unique(y_true)) < 2:
        return None
    
    try:
        y_true_cat = tf.keras.utils.to_categorical(y=y_true, num_classes=num_classes)
        # CRITICAL FIX: Use decision function scores directly
        # For multi-class OvR SVM, decision_function returns scores for each class
        return roc_auc_score(y_true_cat, decision_scores, multi_class='ovo')
    except ValueError:
        return None

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

labels_one_hot = label_encoder.transform(labels)

n = data.shape[0]
outer_kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
inner_kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

all_metrics = dict()

metrics = ["precision", "recall", "f1_score", "accuracy", "auc"]
for m in metrics:
    all_metrics[m] = []
    all_metrics[f"val_{m}"] = []
    all_metrics[f"test_{m}"] = []

# Track best model
best_test_accuracy = 0
best_model = None
best_fold_id = None

for outer_fold, (inner_idx, outer_idx) in enumerate(outer_kf.split(data, labels_one_hot)):

    x_test = data.iloc[outer_idx]
    y_test = labels_one_hot[outer_idx]

    inner_data = data.iloc[inner_idx]
    inner_labels = labels_one_hot[inner_idx]

    for inner_fold, (train_idx, val_idx) in enumerate(inner_kf.split(inner_data, inner_labels)):

        nested_id = f"{str(outer_fold)}_{str(inner_fold)}"

        x_train = inner_data.iloc[train_idx]
        x_val = inner_data.iloc[val_idx]
        
        y_train = inner_labels[train_idx]
        y_val = inner_labels[val_idx]
        
        # CRITICAL FIX: Use decision_function=True for AUC calculation
        # This is implicitly available with SVC
        model = svm.SVC(kernel='linear')
        model.fit(x_train, y_train)

        """
        Training Performance
        """
        # Get discrete predictions for precision/recall/F1/accuracy
        p_train = model.predict(x_train)
        # CRITICAL FIX: Get decision function scores for AUC
        train_decision_scores = model.decision_function(x_train)
        
        train_precision = precision_score(y_train, p_train, average="macro", zero_division=0)
        train_recall = recall_score(y_train, p_train, average="macro", zero_division=0)
        train_f1_score = f1_score(y_train, p_train, average="macro", zero_division=0)
        train_accuracy = accuracy_score(y_train, p_train)
        # CRITICAL FIX: Pass decision scores to AUC calculation
        train_auc = safe_roc_auc_score(y_train, train_decision_scores, num_classes)

        all_metrics["precision"].append(train_precision)
        all_metrics["recall"].append(train_recall)
        all_metrics["f1_score"].append(train_f1_score)
        all_metrics["accuracy"].append(train_accuracy)
        all_metrics["auc"].append(train_auc)

        """
        Validation Performance
        """
        # Get discrete predictions for precision/recall/F1/accuracy
        p_val = model.predict(x_val)
        # CRITICAL FIX: Get decision function scores for AUC
        val_decision_scores = model.decision_function(x_val)
        
        val_precision = precision_score(y_val, p_val, average="macro", zero_division=0)
        val_recall = recall_score(y_val, p_val, average="macro", zero_division=0)
        val_f1_score = f1_score(y_val, p_val, average="macro", zero_division=0)
        val_accuracy = accuracy_score(y_val, p_val)
        # CRITICAL FIX: Pass decision scores to AUC calculation
        val_auc = safe_roc_auc_score(y_val, val_decision_scores, num_classes)

        all_metrics["val_precision"].append(val_precision)
        all_metrics["val_recall"].append(val_recall)
        all_metrics["val_f1_score"].append(val_f1_score)
        all_metrics["val_accuracy"].append(val_accuracy)
        all_metrics["val_auc"].append(val_auc)

        """
        Test Performance
        """ 
        # Get discrete predictions for precision/recall/F1/accuracy
        p_test = model.predict(x_test)
        # CRITICAL FIX: Get decision function scores for AUC
        test_decision_scores = model.decision_function(x_test)
        
        test_precision = precision_score(y_test, p_test, average="macro", zero_division=0)
        test_recall = recall_score(y_test, p_test, average="macro", zero_division=0)
        test_f1_score = f1_score(y_test, p_test, average="macro", zero_division=0)
        test_accuracy = accuracy_score(y_test, p_test)
        # CRITICAL FIX: Pass decision scores to AUC calculation
        test_auc = safe_roc_auc_score(y_test, test_decision_scores, num_classes)

        all_metrics["test_precision"].append(test_precision)
        all_metrics["test_recall"].append(test_recall)
        all_metrics["test_f1_score"].append(test_f1_score)
        all_metrics["test_accuracy"].append(test_accuracy)
        all_metrics["test_auc"].append(test_auc)

        # Track best model based on test accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model = model
            best_fold_id = nested_id

        fold_info = {
            "precision" : train_precision,
            "recall" : train_recall,
            "f1_score" : train_f1_score,
            "accuracy" : train_accuracy,
            "auc" : train_auc,
            "val_precision" : val_precision,
            "val_recall" : val_recall,
            "val_f1_score" : val_f1_score,
            "val_accuracy" : val_accuracy,
            "val_auc" : val_auc,  # FIXED: was incorrectly set to test_auc
            "test_precision" : test_precision,
            "test_recall" : test_recall,
            "test_f1_score" : test_f1_score,
            "test_accuracy" : test_accuracy,
            "test_auc" : test_auc,
        }

        fold_filename = f"{classifier_name}_{nested_id}.pickle"

        with open(fold_filename, 'wb') as handle:
            pickle.dump(
                fold_info,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        
        


            
# Calculate and display mean ± std for each metric across all folds
print("\n" + "="*70)
print("NESTED CROSS-VALIDATION RESULTS")
print("="*70)

# Training set metrics
print("\nTraining Set Metrics:")
print("-" * 70)
for m in metrics:
    values = [v if v is not None else np.nan for v in all_metrics[m]]
    valid_count = np.sum(~np.isnan(values))
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f} (n={valid_count})")

# Validation set metrics
print("\nValidation Set Metrics:")
print("-" * 70)
for m in metrics:
    values = [v if v is not None else np.nan for v in all_metrics[f"val_{m}"]]
    valid_count = np.sum(~np.isnan(values))
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f} (n={valid_count})")

# Test set metrics
print("\nTest Set Metrics:")
print("-" * 70)
for m in metrics:
    values = [v if v is not None else np.nan for v in all_metrics[f"test_{m}"]]
    valid_count = np.sum(~np.isnan(values))
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f} (n={valid_count})")

print("\n" + "="*70)
print(f"Total folds evaluated: {len(all_metrics['accuracy'])}")
print("="*70)

# Save metrics to pickle file
results = {
    "all_metrics": all_metrics
}

with open('svm_nested_cv_results.pickle', 'wb') as handle:
    pickle.dump(
        results,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("\nResults saved to: svm_nested_cv_results.pickle")

# Save best model
if best_model is not None:
    with open(f'{classifier_name}_best_model.pickle', 'wb') as handle:
        pickle.dump({
            'model': best_model,
            'fold_id': best_fold_id,
            'test_accuracy': best_test_accuracy,
            'label_encoder': label_encoder
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nBest model saved to: {classifier_name}_best_model.pickle")
    print(f"Best fold: {best_fold_id}")
    print(f"Test accuracy: {best_test_accuracy:.4f}")
