"""
src/mastitis_features.py
========================
KsheerRaksha-AI (Phase 2 — Visual Udder Abnormality Indicators)

Modular feature extractor calculating non-diagnostic optical and morphological
indicators from cropped udder images.

Terminology Notice:
  Outputs are strictly 'visual udder abnormality indicators' (optical proxies),
  NOT diagnostic biomarkers for intramammary bacterial infection.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional

class VisualUdderFeatureExtractor:
    """
    Computes visual and morphological descriptors from cropped udder imagery:
      1. Erythema / Redness Discoloration Index (HSV space)
      2. Bilateral Left/Right Udder Symmetry Index
      3. Surface Texture Roughness / Vascular Congestion Proxy
      4. Aspect Ratio & Geometric Dimensions
    """
    
    def __init__(self):
        pass
        
    def extract_erythema_index(self, crop: np.ndarray) -> Dict[str, float]:
        """
        Calculates localized skin redness and vascular erythema in HSV and Lab color spaces.
        Healthy udder skin typically exhibits uniform pale/pink or pigmented hues.
        Elevated redness (high a* in Lab, specific H range in HSV) correlates with surface inflammation.
        """
        if crop is None or crop.size == 0:
            return {"erythema_index": 0.0, "red_pixel_ratio": 0.0, "mean_saturation": 0.0}
            
        # Convert BGR to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Red hue wraps around 0/180 in OpenCV (0-10 and 170-180)
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([168, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        total_pixels = crop.shape[0] * crop.shape[1]
        red_pixels = int(np.count_nonzero(red_mask))
        red_ratio = round(red_pixels / max(1, total_pixels), 4)
        
        # Lab color space: 'a' channel represents Green-Red axis
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        l_chan, a_chan, b_chan = cv2.split(lab)
        mean_a = float(np.mean(a_chan))
        
        # Normalized erythema proxy index (scaled 0.0 to 1.0)
        # In OpenCV Lab, a_chan neutral is ~128. Values >135 represent reddish shifts.
        erythema_proxy = float(np.clip((mean_a - 128.0) / 30.0, 0.0, 1.0))
        
        return {
            "erythema_proxy_index": round(erythema_proxy, 4),
            "red_hue_pixel_ratio": red_ratio,
            "mean_saturation": round(float(np.mean(s)) / 255.0, 4),
            "mean_chroma_a": round(mean_a, 2)
        }

    def extract_bilateral_symmetry(self, crop: np.ndarray) -> Dict[str, float]:
        """
        Evaluates left-right visual symmetry across the vertical midline.
        Severe swelling, quarter edema, or teat disparity creates anatomical asymmetry.
        """
        if crop is None or crop.size == 0:
            return {"asymmetry_index": 0.0, "symmetry_confidence": 0.0}
            
        h, w = crop.shape[:2]
        mid = w // 2
        
        left_half = crop[:, :mid]
        right_half = crop[:, mid + (w % 2):]
        
        # Flip right half horizontally for bilateral comparison
        right_flipped = cv2.flip(right_half, 1)
        
        # Resize to matching dimensions if odd width
        min_w = min(left_half.shape[1], right_flipped.shape[1])
        left_half = left_half[:, :min_w]
        right_flipped = right_flipped[:, :min_w]
        
        # Grayscale difference
        g_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
        g_right = cv2.cvtColor(right_flipped, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(g_left, g_right)
        mean_diff = float(np.mean(diff)) / 255.0
        
        # Asymmetry index: 0.0 = perfectly symmetric, 1.0 = highly asymmetric
        asymmetry_index = float(np.clip(mean_diff * 2.5, 0.0, 1.0))
        
        return {
            "asymmetry_index": round(asymmetry_index, 4),
            "bilateral_difference_mean": round(mean_diff, 4)
        }

    def extract_texture_roughness(self, crop: np.ndarray) -> Dict[str, float]:
        """
        Evaluates skin texture roughness using gradient energy / Sobel filter.
        Smooth teat skin has low high-frequency gradient energy;
        hyperkeratosis rings, chapped fissures, and scabs create high texture roughness.
        """
        if crop is None or crop.size == 0:
            return {"texture_roughness_index": 0.0}
            
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_energy = float(np.mean(magnitude))
        
        # Normalized roughness index
        roughness_index = float(np.clip(mean_energy / 80.0, 0.0, 1.0))
        
        return {
            "texture_roughness_index": round(roughness_index, 4),
            "gradient_energy_mean": round(mean_energy, 2)
        }

    def extract_all_indicators(self, crop: np.ndarray) -> Dict[str, Any]:
        """
        Extracts composite dictionary of all visual udder abnormality indicators.
        """
        if crop is None or crop.size == 0:
            return {
                "valid": False,
                "error": "EMPTY_CROP"
            }
            
        h, w = crop.shape[:2]
        erythema = self.extract_erythema_index(crop)
        symmetry = self.extract_bilateral_symmetry(crop)
        texture = self.extract_texture_roughness(crop)
        
        return {
            "valid": True,
            "crop_dimensions": {"width": w, "height": h, "aspect_ratio": round(w / max(1, h), 3)},
            "erythema_indicators": erythema,
            "morphological_symmetry": symmetry,
            "surface_texture": texture,
            "experimental_disclaimer": "Heuristic visual indicators for research demonstration only. Not a veterinary diagnostic test."
        }
