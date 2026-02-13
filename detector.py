"""
Change Detection Module for CSIDC Land Watch
Uses Computer Vision to detect differences between legacy and current maps
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ChangeDetector:
    """
    Detects changes between legacy maps and current satellite imagery
    Highlights land use changes, unauthorized constructions, etc.
    """
    
    def __init__(self):
        self.output_dir = Path("processed")
        self.output_dir.mkdir(exist_ok=True)
        self.change_threshold = 30  # Pixel intensity difference threshold
    
    def detect_changes(
        self, 
        legacy_map_path: str, 
        current_map_path: str,
        sensitivity: float = 0.5
    ) -> Dict:
        """
        Detect changes between two aligned images
        
        Args:
            legacy_map_path: Path to aligned legacy map
            current_map_path: Path to current satellite image
            sensitivity: Detection sensitivity (0.0 to 1.0)
        
        Returns:
            Dictionary with detection results and metrics
        """
        try:
            # Load images
            legacy = cv2.imread(legacy_map_path)
            current = cv2.imread(current_map_path)
            
            if legacy is None or current is None:
                raise ValueError("Could not load one or both images")
            
            # Ensure images are same size
            if legacy.shape != current.shape:
                logger.warning("Images have different sizes, resizing current to match legacy")
                current = cv2.resize(current, (legacy.shape[1], legacy.shape[0]))
            
            logger.info(f"Detecting changes between images of size {legacy.shape}")
            
            # Convert to grayscale for comparison
            legacy_gray = cv2.cvtColor(legacy, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(legacy_gray, current_gray)
            
            # Apply threshold based on sensitivity
            threshold_value = int(self.change_threshold * (1 - sensitivity))
            _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours of changed regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours (noise)
            min_area = 100  # Minimum area in pixels
            significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            logger.info(f"Found {len(significant_contours)} significant change regions")
            
            # Create visualization
            visualization = self._create_visualization(
                legacy, current, thresh, significant_contours
            )
            
            # Save visualization
            vis_filename = f"changes_{Path(legacy_map_path).stem}_vs_{Path(current_map_path).stem}.png"
            vis_path = self.output_dir / vis_filename
            cv2.imwrite(str(vis_path), visualization)
            
            # Calculate change metrics
            total_pixels = legacy_gray.shape[0] * legacy_gray.shape[1]
            changed_pixels = np.count_nonzero(thresh)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            # Generate alerts based on changes
            alerts = self._generate_alerts(significant_contours, legacy.shape)
            
            results = {
                "changes_detected": len(significant_contours) > 0,
                "num_change_regions": len(significant_contours),
                "change_percentage": round(change_percentage, 2),
                "visualization_path": str(vis_path),
                "alerts": alerts,
                "change_regions": self._extract_change_regions(significant_contours, legacy.shape)
            }
            
            logger.info(f"Change detection complete: {change_percentage:.2f}% changed")
            
            return results
        
        except Exception as e:
            logger.error(f"Error detecting changes: {str(e)}")
            raise
    
    def _create_visualization(
        self, 
        legacy: np.ndarray, 
        current: np.ndarray, 
        diff_mask: np.ndarray,
        contours: List
    ) -> np.ndarray:
        """
        Create a visualization showing detected changes
        
        Args:
            legacy: Legacy map image
            current: Current satellite image
            diff_mask: Binary difference mask
            contours: List of change region contours
        
        Returns:
            Visualization image
        """
        # Create side-by-side comparison
        h, w = legacy.shape[:2]
        
        # Create overlay on current image
        overlay = current.copy()
        
        # Draw change regions in red
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        
        # Create colored difference mask
        diff_colored = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)
        diff_colored[diff_mask > 0] = [0, 0, 255]  # Red for changes
        
        # Blend with current image
        blended = cv2.addWeighted(current, 0.7, diff_colored, 0.3, 0)
        
        # Create final visualization: legacy | current | changes
        visualization = np.hstack([legacy, current, blended])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, "Legacy Map", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(visualization, "Current Satellite", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(visualization, "Detected Changes", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return visualization
    
    def _generate_alerts(self, contours: List, image_shape: Tuple) -> List[Dict]:
        """
        Generate alerts based on detected changes
        
        Args:
            contours: List of change region contours
            image_shape: Shape of the image
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Classify change type based on characteristics
            alert_type = "Unknown Change"
            severity = "Medium"
            
            if area > 5000:
                alert_type = "Large-scale Land Use Change"
                severity = "High"
            elif area > 1000:
                alert_type = "Potential Unauthorized Construction"
                severity = "High"
            else:
                alert_type = "Minor Land Modification"
                severity = "Low"
            
            alerts.append({
                "id": i + 1,
                "type": alert_type,
                "severity": severity,
                "area_pixels": int(area),
                "location": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "description": f"{alert_type} detected at region ({x}, {y})"
            })
        
        # Sort by severity
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        alerts.sort(key=lambda a: severity_order[a["severity"]])
        
        return alerts
    
    def _extract_change_regions(self, contours: List, image_shape: Tuple) -> List[Dict]:
        """
        Extract bounding boxes for change regions
        
        Args:
            contours: List of contours
            image_shape: Image dimensions
        
        Returns:
            List of region dictionaries
        """
        regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            regions.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(area),
                "center": [int(x + w/2), int(y + h/2)]
            })
        
        return regions
    
    def compare_with_threshold(
        self, 
        image1_path: str, 
        image2_path: str,
        threshold: int = 30
    ) -> np.ndarray:
        """
        Simple threshold-based comparison
        
        Args:
            image1_path: First image path
            image2_path: Second image path
            threshold: Difference threshold
        
        Returns:
            Binary difference mask
        """
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        diff = cv2.absdiff(img1, img2)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        return mask


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = ChangeDetector()
    
    # Example detection
    # results = detector.detect_changes("legacy_aligned.png", "current_satellite.png")
    # print(f"Changes detected: {results['changes_detected']}")
    # print(f"Number of regions: {results['num_change_regions']}")
    # print(f"Change percentage: {results['change_percentage']}%")
    
    print("ChangeDetector module ready")
