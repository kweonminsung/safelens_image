"""
Configuration constants for SaveLens.
"""

from .models import ReplacementMethod

# Model Settings
DETECTION_MODEL = "gemini-3-flash-preview"  # Model for PII and face detection
IMAGEN_MODEL = "gemini-2.5-flash-image"  # Model for image generation

# Detection Settings
MIN_CONFIDENCE = 0.8

# Default Anonymization Methods
DEFAULT_FACE_METHOD = ReplacementMethod.BLUR
DEFAULT_TEXT_METHOD = ReplacementMethod.GENERATE

# Image Generator Settings
MASK_PADDING = 10  # Padding around masked regions in pixels
