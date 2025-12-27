"""
Configuration constants for SafeLens.
"""

from .models import ReplacementMethod

# Model Settings
DETECTION_MODEL = "gemini-3-flash-preview"  # Model for PII and face detection
# IMAGEN_MODEL = "gemini-2.5-flash-image"  # Model for image generation
IMAGEN_MODEL = "gemini-3-pro-image-preview"  # Model for image generation

# Detection Settings
MIN_FACE_CONFIDENCE = 0.7  # Minimum confidence for face detection
MIN_TEXT_CONFIDENCE = 0.8  # Minimum confidence for text PII detection

# Default Anonymization Methods
DEFAULT_FACE_METHOD = ReplacementMethod.BLUR
DEFAULT_TEXT_METHOD = ReplacementMethod.GENERATE

# Image Generator Settings
MASK_PADDING = 10  # Padding around masked regions in pixels

# Image Quality Settings
THUMBNAIL_MAX_WIDTH = 400  # Maximum width for low-quality thumbnails (height scales proportionally)
