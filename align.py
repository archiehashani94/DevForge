"""
Image Alignment Module for CSIDC Land Watch
Performs geo-rectification using OpenCV affine transformations
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


class ImageAligner:
    """
    Aligns legacy map images to GPS coordinates using affine transformation
    """
    
    def __init__(self):
        self.output_dir = Path("processed")
        self.output_dir.mkdir(exist_ok=True)
    
    def align_image(
        self, 
        image_path: str, 
        bounds: Dict[str, float],
        reference_points: Optional[List[Tuple[Tuple[int, int], Tuple[float, float]]]] = None
    ) -> str:
        """
        Align an image to GPS coordinates using affine transformation
        
        Args:
            image_path: Path to the input image
            bounds: Dictionary with lat_min, lat_max, lon_min, lon_max
            reference_points: Optional list of (image_coords, gps_coords) tuples
                            for manual control point matching
        
        Returns:
            Path to the aligned/geo-rectified image
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape[:2]
            logger.info(f"Loaded image: {width}x{height}")
            
            # Define source points (corners of the original image)
            src_points = np.float32([
                [0, 0],                    # Top-left
                [width - 1, 0],            # Top-right
                [0, height - 1],           # Bottom-left
                [width - 1, height - 1]    # Bottom-right
            ])
            
            # If custom reference points provided, use them
            if reference_points and len(reference_points) >= 3:
                logger.info("Using custom reference points for alignment")
                src_points = np.float32([pt[0] for pt in reference_points[:4]])
                # Convert GPS to pixel coordinates (simplified projection)
                dst_points = self._gps_to_pixels(
                    [pt[1] for pt in reference_points[:4]], 
                    bounds, 
                    width, 
                    height
                )
            else:
                # Map image corners to GPS bounds
                # Assuming image is already roughly aligned (top = north)
                dst_points = np.float32([
                    [0, 0],                    # Top-left (lat_max, lon_min)
                    [width - 1, 0],            # Top-right (lat_max, lon_max)
                    [0, height - 1],           # Bottom-left (lat_min, lon_min)
                    [width - 1, height - 1]    # Bottom-right (lat_min, lon_max)
                ])
            
            # Calculate affine transformation matrix
            # For more complex transformations, use cv2.getPerspectiveTransform
            if len(src_points) == 3:
                matrix = cv2.getAffineTransform(src_points[:3], dst_points[:3])
                aligned = cv2.warpAffine(image, matrix, (width, height))
            else:
                # Use perspective transform for 4+ points
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                aligned = cv2.warpPerspective(image, matrix, (width, height))
            
            # Save the aligned image
            output_filename = f"aligned_{Path(image_path).name}"
            output_path = self.output_dir / output_filename
            cv2.imwrite(str(output_path), aligned)
            
            logger.info(f"Aligned image saved to: {output_path}")
            
            # Also save metadata
            self._save_metadata(output_path, bounds, matrix)
            
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error aligning image: {str(e)}")
            raise
    
    def _gps_to_pixels(
        self, 
        gps_coords: List[Tuple[float, float]], 
        bounds: Dict[str, float],
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Convert GPS coordinates to pixel coordinates
        
        Args:
            gps_coords: List of (lat, lon) tuples
            bounds: GPS bounding box
            width, height: Image dimensions
        
        Returns:
            Array of pixel coordinates
        """
        lat_range = bounds["lat_max"] - bounds["lat_min"]
        lon_range = bounds["lon_max"] - bounds["lon_min"]
        
        pixel_coords = []
        for lat, lon in gps_coords:
            # Normalize to 0-1 range
            x_norm = (lon - bounds["lon_min"]) / lon_range
            y_norm = (bounds["lat_max"] - lat) / lat_range  # Inverted Y axis
            
            # Scale to image dimensions
            x = int(x_norm * width)
            y = int(y_norm * height)
            
            pixel_coords.append([x, y])
        
        return np.float32(pixel_coords)
    
    def _save_metadata(self, image_path: Path, bounds: Dict[str, float], matrix: np.ndarray):
        """Save alignment metadata for future reference"""
        metadata_path = image_path.with_suffix('.meta.txt')
        
        with open(metadata_path, 'w') as f:
            f.write(f"Bounds: {bounds}\n")
            f.write(f"Transformation Matrix:\n{matrix}\n")
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def extract_features(self, image_path: str) -> Tuple[List, np.ndarray]:
        """
        Extract keypoints and descriptors for feature-based alignment
        Useful for automatic alignment when reference points are unknown
        
        Args:
            image_path: Path to image
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Use ORB (Oriented FAST and Rotated BRIEF) detector
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        logger.info(f"Extracted {len(keypoints)} keypoints from {image_path}")
        
        return keypoints, descriptors
    
    def match_and_align(
        self, 
        legacy_image_path: str, 
        reference_image_path: str,
        bounds: Dict[str, float]
    ) -> str:
        """
        Automatically align legacy image to reference using feature matching
        
        Args:
            legacy_image_path: Path to legacy map
            reference_image_path: Path to current satellite image
            bounds: GPS bounds for the reference image
        
        Returns:
            Path to aligned image
        """
        # Extract features from both images
        kp1, desc1 = self.extract_features(legacy_image_path)
        kp2, desc2 = self.extract_features(reference_image_path)
        
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use top matches to compute transformation
        num_matches = min(50, len(matches))
        logger.info(f"Using top {num_matches} matches for alignment")
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:num_matches]])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:num_matches]])
        
        # Find homography
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp legacy image
        legacy_img = cv2.imread(legacy_image_path)
        reference_img = cv2.imread(reference_image_path)
        h, w = reference_img.shape[:2]
        
        aligned = cv2.warpPerspective(legacy_img, matrix, (w, h))
        
        # Save result
        output_filename = f"auto_aligned_{Path(legacy_image_path).name}"
        output_path = self.output_dir / output_filename
        cv2.imwrite(str(output_path), aligned)
        
        logger.info(f"Auto-aligned image saved to: {output_path}")
        
        return str(output_path)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    aligner = ImageAligner()
    
    # Example: Align a legacy map to Nava Raipur bounds
    bounds = {
        "lat_min": 21.10,
        "lat_max": 21.22,
        "lon_min": 81.70,
        "lon_max": 81.86
    }
    
    # aligner.align_image("legacy_map.png", bounds)
    print("ImageAligner module ready")
