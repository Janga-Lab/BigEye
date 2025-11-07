"""
Standalone training script for DeepLabV3+ retinal lesion segmentation.
"""

import os
import sys
import argparse
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold

# --- Import from local files ---

from deeplab import DeeplabV3Plus
from image_utils import read_image_or_mask, apply_clahe

# --- Keras Custom Metric ---

class F1ScoreSegmentation(tf.keras.metrics.Metric):
    """
    Custom F1 Score metric for segmentation.
    Computes the macro-average F1 score across all classes.
    """
    def __init__(self, num_classes=7, name='f1_score', **kwargs):
        super(F1ScoreSegmentation, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.f1_score = self.add_weight(name='f1', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_pred.shape) == 4 and y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_pred = tf.one_hot(y_pred, depth=self.num_classes)
        
        if len(y_true.shape) == 4 and y_true.shape[-1] != self.num_classes:
             if y_true.shape[-1] == 1:
                 y_true = tf.squeeze(y_true, axis=-1)
             y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)

        y_true = tf.cast(y_true, y_pred.dtype)

        tp = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[0, 1, 2])
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[0, 1, 2])

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1 = 2 * precision * recall / (precision + recall + K.epsilon())
        f1_mean = tf.reduce_mean(f1)
        
        if not tf.math.is_nan(f1_mean):
            self.f1_score.assign_add(f1_mean)
            self.count.assign_add(1)

    def result(self):
        return self.f1_score / (self.count + K.epsilon())

    def reset_state(self):
        self.f1_score.assign(0)
        self.count.assign(0)

# --- Keras Custom Layer ---

class OneHotEncode(tf.keras.layers.Layer):
    """
    Custom layer to one-hot encode masks within the tf.data pipeline.
    """
    def __init__(self, depth=7):
        super().__init__()
        self.depth = depth

    def call(self, images, masks, weights):
        images = tf.identity(images)
        weights = tf.identity(weights)
        masks = tf.cast(masks, dtype=tf.int32)
        masks = tf.one_hot(masks, depth=self.depth)
        return images, masks, weights

# --- Data Augmentation ---

# We define a single, stateful augmentation model.
# This ensures the same random transformations are applied.
# We use "nearest" interpolation, which is critical for masks.
augmentation_model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal"),
    tf.keras.layers.RandomFlip(mode="vertical"),
    tf.keras.layers.RandomZoom(
        (-0.1, 0.25),
        fill_mode="constant",
        fill_value=0,
        interpolation="nearest",
    )
])

@tf.function
def augment(images, masks, weights):
    """
    Applies identical geometric augmentations to image, mask, and weights.
    """
    # Add channel dim to 2D tensors
    masks = tf.expand_dims(masks, axis=-1)
    weights = tf.expand_dims(weights, axis=-1)
    
    # Cast all to float32 for augmentation model
    images = tf.cast(images, tf.float32)
    masks = tf.cast(masks, tf.float32)
    weights = tf.cast(weights, tf.float32)
    
    # Concatenate on channel axis
    combined = tf.concat([images, masks, weights], axis=-1)
    
    # Apply augmentation to combined tensor
    combined_aug = augmentation_model(combined)
    
    # Split back
    images_aug = combined_aug[..., :3]
    masks_aug = combined_aug[..., 3:4]
    weights_aug = combined_aug[..., 4:]
    
    # Squeeze channel dim from 3D tensors
    masks_aug = tf.squeeze(masks_aug, axis=-1)
    weights_aug = tf.squeeze(weights_aug, axis=-1)
    
    return images_aug, masks_aug, weights_aug


# --- Data Loading Functions ---

def load_metadata_mappings(metadata_path: str):
    """
    Loads label mappings, class weights, and lesion names from the metadata TSV.
    """
    print(f"Loading metadata from: {metadata_path}")
    try:
        df = pd.read_csv(metadata_path, sep='\t')
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        sys.exit(1)

    if 'weight' not in df.columns:
        print(f"Error: Metadata file '{metadata_path}' must contain a 'weight' column.")
        sys.exit(1)

    original_to_encoded = dict(zip(df['grayscale_value'], df['one_hot_value']))
    encoded_to_original = dict(zip(df['one_hot_value'], df['grayscale_value']))
    weight_map_dict = dict(zip(df['one_hot_value'], df['weight']))
    encoded_to_name_map = dict(zip(df['one_hot_value'], df['lesion_name']))
    num_classes = len(df)
    
    print(f"Found {num_classes} classes.")
    
    return (
        original_to_encoded, 
        encoded_to_original, 
        weight_map_dict, 
        num_classes, 
        encoded_to_name_map
    )

def _load_dataset_file(tsv_path: str):
    """
    Loads the dataset from the TSV file.
    Returns a list of dictionaries.
    """
    print(f"Loading dataset from: {tsv_path}")
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {tsv_path}")
        sys.exit(1)

    required_cols = ['image_path', 'mask_path']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: TSV file must contain 'image_path' and 'mask_path' columns.")
        sys.exit(1)

    data = []
    for _, row in df.iterrows():
        image_path = pathlib.Path(row['image_path'])
        mask_path = pathlib.Path(row['mask_path']) if pd.notna(row['mask_path']) else None
        
        if not image_path.is_file():
            print(f"Warning: Image file not found, skipping: {image_path}")
            continue
        if mask_path and not mask_path.is_file():
            print(f"Warning: Mask file not found, skipping: {mask_path}")
            continue

        data.append({"image": image_path, "mask": mask_path})
        
    print(f"Loaded {len(data)} valid image/mask pairs.")
    return np.array(data)

# --- Data Processing Class ---

class DataProcessor:
    """
    Handles processing of a single image/mask pair for the tf.data pipeline.
    """
    def __init__(self, original_to_encoded, weight_map_dict, image_size=512):
        self.original_to_encoded = original_to_encoded
        self.weight_map_dict = weight_map_dict
        self.image_size = (image_size, image_size)
        self.image_shape_1d = (image_size, image_size)
        self.background_encoded_val = original_to_encoded.get(0, 0) # Default to 0

    def create_weight_map(self, encoded_mask):
        """Creates a pixel-wise weight map from an encoded mask."""
        weights = np.ones_like(encoded_mask).astype(np.float32)
        for encoded_val, weight in self.weight_map_dict.items():
            weights[encoded_mask == encoded_val] = weight
        return weights

    def process(self, data_point):
        """Processes one data point (image, mask)."""
        # 1. Process Image
        image = read_image_or_mask(
            data_point["image"].as_posix(),
            resize_dims=self.image_size,
        )
        image = apply_clahe(image)
        image = image * (1 / 255.0) # Normalize to [0, 1]

        # 2. Process Mask
        if data_point["mask"] is None:
            mask_raw = np.zeros(self.image_shape_1d, dtype=np.uint8)
        else:
            mask_raw = read_image_or_mask(
                data_point["mask"].as_posix(),
                resize_dims=self.image_size,
                as_grayscale=True
            )

        # 3. Encode Mask
        encoded_mask = np.full(self.image_shape_1d, self.background_encoded_val, dtype=np.int32)
        for original_val, encoded_val in self.original_to_encoded.items():
            encoded_mask[mask_raw == original_val] = encoded_val

        # 4. Create Weight Map
        weight_map = self.create_weight_map(encoded_mask)
        
        return (
            image.astype(np.float32), 
            encoded_mask.astype(np.float32), 
            weight_map.astype(np.float32)
        )

# --- TF Dataset Creation ---

def tensorflow_dataset_handler(dataset, processor: DataProcessor):
    """Generator function for tf.data.Dataset."""
    def generator():
        for data_point in dataset:
            yield processor.process(data_point)
    return generator

def create_tf_dataset(
    dataset: np.ndarray, 
    processor: DataProcessor, 
    batch_size: int, 
    num_classes: int, 
    augment_data: bool = False
):
    """
    Creates a pre-fetched, batched, and one-hot encoded tf.data.Dataset.
    Optionally applies augmentation.
    """
    
    output_signature = (
        tf.TensorSpec(shape=(processor.image_size[0], processor.image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(processor.image_size[0], processor.image_size[1]), dtype=tf.float32),
        tf.TensorSpec(shape=(processor.image_size[0], processor.image_size[1]), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(
            tensorflow_dataset_handler(dataset, processor),
            output_signature=output_signature,
        )
    
    # Cache the raw, processed data before augmentation
    ds = ds.cache()

    # Apply augmentation *only* if specified (i.e., for training data)
    if augment_data:
        print("Note: Applying data augmentation (flips, zoom).")
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply one-hot encoding, batching, and prefetching
    return (
        ds
        .map(OneHotEncode(depth=num_classes), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

# --- Metrics Setup ---

def get_metrics(num_classes: int, encoded_to_name_map: dict):
    """
    Initializes all Keras metrics, using lesion names for IoU.
    """
    metrics = [
        F1ScoreSegmentation(num_classes=num_classes, name="f1_score"),
        tf.keras.metrics.OneHotIoU(
            num_classes=num_classes, 
            name="mean_iou"
        )
    ]
    
    for encoded_val, lesion_name in encoded_to_name_map.items():
        metric_name = str(lesion_name).lower().replace(' ', '_') + "_iou"
        metrics.append(
            tf.keras.metrics.OneHotIoU(
                num_classes=num_classes,
                target_class_ids=[int(encoded_val)],
                name=metric_name
            )
        )
    
    return metrics

# --- Main Training Function ---

def train(args):
    """
    Main training and cross-validation loop.
    """
    
    # 1. Load Mappings and Data
    (
        original_to_encoded, 
        encoded_to_original, 
        weight_map_dict, 
        num_classes, 
        encoded_to_name_map
    ) = load_metadata_mappings(args.metadata_tsv)
    
    all_data = _load_dataset_file(args.input_tsv)
    
    if len(all_data) == 0:
        print("No data to train on. Exiting.")
        return

    # 2. Setup K-Fold
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    best_val_iou = -1.0
    best_model_path = "best_deeplab_weights.weights.h5"
    
    # 3. Instantiate DataProcessor
    processor = DataProcessor(
        original_to_encoded, 
        weight_map_dict, 
        args.image_size
    )

    # 4. K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n--- Fold {fold+1}/{args.k_folds} ---")
        
        train_data_points = all_data[train_idx]
        val_data_points = all_data[val_idx]
        
        print(f"Training on {len(train_data_points)} samples, validating on {len(val_data_points)} samples.")

        # 5. Create TF Datasets
        print("Creating TF Datasets...")
        # Augmentation is enabled for training data
        train_ds = create_tf_dataset(
            train_data_points, 
            processor, 
            args.batch_size, 
            num_classes, 
            augment_data=args.augment
        )
        # Augmentation is DISABLED for validation data
        val_ds = create_tf_dataset(
            val_data_points, 
            processor, 
            args.batch_size, 
            num_classes, 
            augment_data=False
        )

        # 6. Initialize Model
        print("Initializing new model for fold...")
        model = DeeplabV3Plus(num_classes=num_classes, image_size=args.image_size)

        # 7. Compile Model
        metrics = get_metrics(num_classes, encoded_to_name_map)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            weighted_metrics=metrics
        )

        # 8. Callbacks
        fold_checkpoint_path = f"temp_fold_{fold}_best.weights.h5"
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=fold_checkpoint_path,
            monitor='val_mean_iou',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            verbose=1,
            restore_best_weights=False
        )

        # 9. Fit Model
        print(f"Starting training for fold {fold+1}...")
        history = model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data=val_ds,
            callbacks=[checkpoint_cb, early_stop_cb]
        )

        # 10. Evaluate and Save Overall Best
        current_fold_best_iou = max(history.history['val_mean_iou'])
        print(f"Fold {fold+1} Best Validation Mean IoU: {current_fold_best_iou:.5f}")

        if current_fold_best_iou > best_val_iou:
            print(f"New overall best model found! (Mean IoU: {current_fold_best_iou:.5f} > {best_val_iou:.5f})")
            print(f"Saving weights to {best_model_path}")
            best_val_iou = current_fold_best_iou
            if os.path.exists(best_model_path):
                 os.remove(best_model_path)
            os.rename(fold_checkpoint_path, best_model_path)
        else:
            if os.path.exists(fold_checkpoint_path):
                os.remove(fold_checkpoint_path)

        K.clear_session()

    print("\n--- Training Complete ---")
    if best_val_iou >= 0:
        print(f"Best model weights saved to {best_model_path} with Val Mean IoU: {best_val_iou:.5f}")
    else:
        print("Training finished, but no model was saved (best val_mean_iou remained <= 0).")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3+ for Retinal Lesion Segmentation."
    )
    
    parser.add_argument(
        "--input_tsv", 
        type=str, 
        required=True,
        help="Path to the TSV file with columns [image_id, image_path, mask_path, dr_stage]."
    )
    
    parser.add_argument(
        "--metadata_tsv", 
        type=str, 
        required=True,
        help="Path to the retinal_lesion_metadata.tsv file for class mappings and weights."
    )
    
    parser.add_argument(
        "--k_folds", 
        type=int, 
        default=5,
        help="Number of folds for K-Fold Cross-Validation."
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of epochs to train per fold."
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size."
    )
    
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=512,
        help="Size to resize images and masks (e.g., 512 for 512x512)."
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="Learning rate for the Adam optimizer."
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        help="If set, applies random flips and zoom augmentations to the training data."
    )

    args = parser.parse_args()
    
    train(args)
