import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, StratifiedKFold



random_state = 10
classifier_name = "nn"

def standardize_data(data: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
    """ Standardize data using StandardScaler """
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    return data_standardized



def create_model(
        k: 5,  # number of classes
        n_samples,
        dropout_rate: float = 0.2  # rate for dropout layers
    ) -> Sequential:
    """ Create Sequential model for multiclass classifier """
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=12)) # 70
    #if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation='relu'))
    #if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
    model.add(Dense(k, activation='softmax'))


    initial_learning_rate = 1e-3
    final_learning_rate = 1e-4
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/500)
    steps_per_epoch = int(n_samples/32)



    lr_exponential_decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = steps_per_epoch,
        decay_rate = learning_rate_decay_factor,
        staircase = True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_exponential_decay_schedule)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name = "accuracy"),
        tf.keras.metrics.Recall(name = "recall"),
        tf.keras.metrics.Precision(name = "precision"),
        tf.keras.metrics.AUC(from_logits = False, name = "auc")

    ]

    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = metrics
    )

    
    return model


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)










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
labels_flat = np.unique(labels_one_hot)

print(dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))


n = data.shape[0]
outer_kf = StratifiedKFold(n_splits = 10, random_state = 0, shuffle = True)
inner_kf = StratifiedKFold(n_splits = 10, random_state = 0, shuffle = True)

# Initialize metrics dictionary
all_metrics = dict()
metrics_list = ["loss", "precision", "recall", "f1_score", "accuracy", "auc"]
for m in metrics_list:
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
    y_test = tf.keras.utils.to_categorical(
        y = y_test,
        num_classes = num_classes
    )

    inner_data = data.iloc[inner_idx]
    inner_labels = labels_one_hot[inner_idx]


    for inner_fold, (train_idx, val_idx) in enumerate(inner_kf.split(inner_data, inner_labels)):

        nested_id = f"{str(outer_fold)}_{str(inner_fold)}"

        x_train = inner_data.iloc[train_idx]
        x_val = inner_data.iloc[val_idx]
        
        y_train = inner_labels[train_idx]
        y_val = inner_labels[val_idx]

        y_train = tf.keras.utils.to_categorical(
            y = y_train,
            num_classes = num_classes
        )

        y_val = tf.keras.utils.to_categorical(
            y = y_val,
            num_classes = num_classes
        )

        n_samples = len(x_train)
        model = create_model(num_classes, n_samples)
        
        history = model.fit(
            x_train,
            y_train,
            validation_data = (x_val, y_val),
            epochs = 300,
            batch_size = 32,
            verbose = 0  # Suppress epoch-by-epoch output
        )

        # Find epoch with best validation accuracy
        val_acc = history.history['val_accuracy']
        best_epoch = np.argmax(val_acc)
        
        # Get training metrics at best epoch
        train_loss = history.history['loss'][best_epoch]
        train_precision = history.history['precision'][best_epoch]
        train_recall = history.history['recall'][best_epoch]
        train_f1_score = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-7)
        train_accuracy = history.history['accuracy'][best_epoch]
        train_auc = history.history['auc'][best_epoch]
        
        
        all_metrics["loss"].append(train_loss)
        all_metrics["recall"].append(train_recall)
        all_metrics["precision"].append(train_precision)
        all_metrics["accuracy"].append(train_accuracy)
        all_metrics["auc"].append(train_auc)
        all_metrics["f1_score"].append(train_f1_score)
        
        
        # Get validation metrics at best epoch
        val_loss = history.history['val_loss'][best_epoch]
        val_precision = history.history['val_precision'][best_epoch]
        val_recall = history.history['val_recall'][best_epoch]
        val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
        val_accuracy = history.history['val_accuracy'][best_epoch]
        val_auc = history.history['val_auc'][best_epoch]

        all_metrics["val_loss"].append(val_loss)
        all_metrics["val_recall"].append(val_recall)
        all_metrics["val_precision"].append(val_precision)
        all_metrics["val_accuracy"].append(val_accuracy)
        all_metrics["val_auc"].append(val_auc)
        all_metrics["val_f1_score"].append(val_f1_score)

        # Evaluate on test set
        test_performance = model.evaluate(
            x_test, y_test, batch_size = 32, verbose = 0
        )
        
        # test_performance is [loss, accuracy, recall, precision, auc]
        test_loss = test_performance[0]
        test_precision = test_performance[3]
        test_recall = test_performance[2]
        test_f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
        test_accuracy = test_performance[1]
        test_auc = test_performance[4]

        all_metrics["test_loss"].append(test_loss)
        all_metrics["test_recall"].append(test_recall)
        all_metrics["test_precision"].append(test_precision)
        all_metrics["test_accuracy"].append(test_accuracy)
        all_metrics["test_auc"].append(test_auc)
        all_metrics["test_f1_score"].append(test_f1_score)

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
            "loss" : train_loss,
            "val_precision" : val_precision,
            "val_recall" : val_recall,
            "val_f1_score" : val_f1_score,
            "val_accuracy" : val_accuracy,
            "val_auc" : val_auc,
            "val_loss" : val_loss,
            "test_precision" : test_precision,
            "test_recall" : test_recall,
            "test_f1_score" : test_f1_score,
            "test_accuracy" : test_accuracy,
            "test_auc" : test_auc,
            "test_loss" : test_loss,
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

# Training set metrics (at best validation epoch)
print("\nTraining Set Metrics (at best validation epoch):")
print("-" * 70)
for m in metrics_list:
    values = all_metrics[m]
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

# Validation set metrics (at best validation epoch)
print("\nValidation Set Metrics (at best validation epoch):")
print("-" * 70)
for m in metrics_list:
    values = all_metrics[f"val_{m}"]
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

# Test set metrics
print("\nTest Set Metrics:")
print("-" * 70)
for m in metrics_list:
    values = all_metrics[f"test_{m}"]
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{m.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")

print("\n" + "="*70)
print(f"Total folds evaluated: {len(all_metrics['accuracy'])}")
print("="*70)

# Save metrics to pickle file
results = {
    "metrics_list": metrics_list,
    "all_metrics": all_metrics
}

with open('nn_nested_cv_results.pickle', 'wb') as handle:
    pickle.dump(
        results,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("\nResults saved to: nn_nested_cv_results.pickle")

# Save best model
if best_model is not None:
    # Save as h5 file
    h5_filename = f'{classifier_name}_best_model.h5'
    best_model.save(h5_filename)
    
    # Save metadata as pickle
    with open(f'{classifier_name}_best_model.pickle', 'wb') as handle:
        pickle.dump({
            'fold_id': best_fold_id,
            'test_accuracy': best_test_accuracy,
            'label_encoder': label_encoder,
            'num_classes': num_classes
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nBest model saved to: {h5_filename}")
    print(f"Model metadata saved to: {classifier_name}_best_model.pickle")
    print(f"Best fold: {best_fold_id}")
    print(f"Test accuracy: {best_test_accuracy:.4f}")
