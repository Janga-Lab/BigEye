"""
Retinal Lesion Feature Extraction Pipeline

Given a directory of retinal fundus images:
1. Runs segmentation model to identify lesions
2. Extracts features (area ratios, contour counts) for each lesion type
3. Outputs consolidated results to TSV file
4. Optionally saves segmentation masks
"""
import argparse
import pathlib
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, Optional

from image_utils import (
    read_image, apply_clahe, extract_fundus_profile, 
    count_contours, normalize_image
)
from deeplab import DeepLabV3Plus


class RetinalLesionFeatureExtractor:
    """Extract features from retinal images using segmentation model."""
    
    def __init__(self, model_weights_path: str, metadata_path: str, image_size: int = 512):
        """
        Initialize feature extractor.
        
        Args:
            model_weights_path: Path to model weights (.hdf5 or .h5)
            metadata_path: Path to lesion metadata TSV
            image_size: Size to resize images for model input
        """
        self.image_size = image_size
        self.metadata = self._load_metadata(metadata_path)
        self.model = self._load_model(model_weights_path)
        
    def _load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """Load and process lesion metadata."""
        df = pd.read_csv(metadata_path, sep='\t')
        df['lesion_name'] = df['lesion_name'].str.strip()
        # Filter out background
        df = df[df['one_hot_value'] != 0].copy()
        return df
    
    def _load_model(self, weights_path: str) -> tf.keras.Model:
        """Load segmentation model with weights."""
        num_classes = len(self.metadata) + 1  # +1 for background
        model = DeepLabV3Plus(num_classes, image_size=self.image_size)
        model.load_weights(weights_path)
        return model
    
    def preprocess_image(self, image_path: pathlib.Path) -> np.ndarray:
        """
        Load and preprocess image for model input.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image ready for model
        """
        # Read and resize
        image = read_image(image_path, grayscale=False, 
                          resize_dims=(self.image_size, self.image_size))
        
        # Apply CLAHE enhancement
        image = apply_clahe(image)
        
        # Normalize
        image = normalize_image(image, max_value=255.0)
        
        return image
    
    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate segmentation mask for image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Segmentation mask with class labels
        """
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Predict
        prediction = self.model.predict(image_batch, verbose=0)
        
        # Convert to class labels
        mask = np.argmax(prediction, axis=-1)[0, :, :]
        
        return mask.astype(np.uint8)
    
    def extract_features(self, image_path: pathlib.Path, mask: np.ndarray) -> Dict:
        """
        Extract lesion features from segmentation mask.
        
        Args:
            image_path: Path to original image (for fundus profile)
            mask: Segmentation mask
            
        Returns:
            Dictionary of features
        """
        features = {'image_id': image_path.stem}
        
        # Get fundus profile for normalization
        gray_image = read_image(image_path, grayscale=True, 
                               resize_dims=(self.image_size, self.image_size))
        fundus_mask = extract_fundus_profile(gray_image)
        fundus_area = np.sum(fundus_mask == 255)
        
        if fundus_area == 0:
            # If no valid fundus detected, use whole image
            fundus_area = mask.shape[0] * mask.shape[1]
        
        # Extract features for each lesion type
        for _, row in self.metadata.iterrows():
            lesion_name = row['lesion_name'].replace(' ', '_')
            class_value = row['one_hot_value']
            
            # Count pixels for this lesion type
            lesion_pixels = np.sum(mask == class_value)
            
            # Calculate area ratio
            area_ratio = lesion_pixels / fundus_area if fundus_area > 0 else 0
            features[f'{lesion_name}_area_ratio'] = area_ratio
            
            # Count separate contours
            contour_count = count_contours(mask, class_value)
            features[f'{lesion_name}_contour_count'] = contour_count
        
        return features
    
    def process_directory(self, 
                         input_dir: pathlib.Path,
                         output_tsv: pathlib.Path,
                         save_masks: bool = False,
                         mask_output_dir: Optional[pathlib.Path] = None) -> pd.DataFrame:
        """
        Process all images in directory and extract features.
        
        Args:
            input_dir: Directory containing input images
            output_tsv: Path for output TSV file
            save_masks: Whether to save segmentation masks
            mask_output_dir: Directory to save masks (required if save_masks=True)
            
        Returns:
            DataFrame with extracted features
        """
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_dir.glob(ext))
            image_paths.extend(input_dir.glob(ext.upper()))
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"Found {len(image_paths)} images to process")
        
        # Create mask output directory if needed
        if save_masks:
            if mask_output_dir is None:
                raise ValueError("mask_output_dir must be provided when save_masks=True")
            mask_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        all_features = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Preprocess and predict
                image = self.preprocess_image(image_path)
                mask = self.predict_mask(image)
                
                # Extract features
                features = self.extract_features(image_path, mask)
                all_features.append(features)
                
                # Save mask if requested
                if save_masks:
                    mask_path = mask_output_dir / f"{image_path.stem}_mask.png"
                    cv2.imwrite(str(mask_path), mask)
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_features)
        df.to_csv(output_tsv, sep='\t', index=False)
        
        print(f"\nProcessed {len(df)} images successfully")
        print(f"Results saved to: {output_tsv}")
        if save_masks:
            print(f"Masks saved to: {mask_output_dir}")
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Extract retinal lesion features from fundus images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input retinal images'
    )
    
    parser.add_argument(
        '--output-tsv',
        type=str,
        required=True,
        help='Path for output TSV file with features'
    )
    
    parser.add_argument(
        '--model-weights',
        type=str,
        default='retinal_lesion_segmentation.hdf5',
        help='Path to model weights file'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        default='retinal_lesion_metadata.tsv',
        help='Path to lesion metadata TSV file'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        default=512,
        help='Size to resize images for model input'
    )
    
    parser.add_argument(
        '--save-masks',
        action='store_true',
        help='Save segmentation masks to separate directory'
    )
    
    parser.add_argument(
        '--mask-output-dir',
        type=str,
        default=None,
        help='Directory to save segmentation masks (required if --save-masks is used)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = pathlib.Path(args.input_dir)
    output_tsv = pathlib.Path(args.output_tsv)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    mask_output_dir = None
    if args.save_masks:
        if args.mask_output_dir is None:
            raise ValueError("--mask-output-dir must be provided when --save-masks is used")
        mask_output_dir = pathlib.Path(args.mask_output_dir)
    
    # Create output directory if needed
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor and process
    extractor = RetinalLesionFeatureExtractor(
        model_weights_path=args.model_weights,
        metadata_path=args.metadata,
        image_size=args.image_size
    )
    
    # Process images
    df = extractor.process_directory(
        input_dir=input_dir,
        output_tsv=output_tsv,
        save_masks=args.save_masks,
        mask_output_dir=mask_output_dir
    )
    
    # Display summary statistics
    print("\n" + "="*60)
    print("Feature Extraction Summary")
    print("="*60)
    
    # Show mean values for each feature
    feature_cols = [col for col in df.columns if col != 'image_id']
    summary = df[feature_cols].describe().loc[['mean', 'std']].T
    print(summary.to_string())


if __name__ == '__main__':
    main()
