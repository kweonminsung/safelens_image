"""
Anonymization module for applying various replacement methods to detected regions.
Supports both polygon and bounding box regions.
"""
import io
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import cv2
from .models import (
    BoundingBox,
    ReplacementMethod,
    PIIDetection,
    FaceDetection,
    ReplacementRequest,
)


class Anonymizer:
    """Applies anonymization techniques to image regions."""
    
    def __init__(
        self,
        default_face_method: ReplacementMethod = ReplacementMethod.BLUR,
        default_text_method: ReplacementMethod = ReplacementMethod.MASK
    ):
        """Initialize anonymizer with default settings.
        
        Args:
            default_face_method: Default method for face anonymization (BLUR or INPAINT recommended)
            default_text_method: Default method for text PII anonymization (MASK or REDACT recommended)
        """
        self.default_face_method = default_face_method
        self.default_text_method = default_text_method
        
        self.emoji_map = {
            'default': '😊',
            'smile': '😊',
            'neutral': '😐',
            'cool': '😎',
            'robot': '🤖',
        }
    
    def anonymize_image(
        self,
        image: Image.Image,
        replacements: List[Tuple[BoundingBox, ReplacementMethod, Optional[str]]]
    ) -> Image.Image:
        """
        Apply multiple anonymization replacements to an image.
        
        Args:
            image: Original PIL Image
            replacements: List of (bbox, method, custom_data) tuples
            
        Returns:
            Anonymized PIL Image
        """
        # Work on a copy
        result = image.copy()
        
        for region, method, custom_data in replacements:
            result = self._apply_replacement(result, region, method, custom_data)
        
        return result
    
    def _apply_replacement(
        self,
        image: Image.Image,
        region: BoundingBox,
        method: ReplacementMethod,
        custom_data: Optional[str] = None
    ) -> Image.Image:
        """Apply a single replacement to an image region."""
        
        if method == ReplacementMethod.MASK:
            return self._mask_text(image, region)
        elif method == ReplacementMethod.SYNTHETIC_TEXT:
            return self._replace_with_synthetic_text(image, region, custom_data)
        elif method == ReplacementMethod.REDACT:
            return self._redact_region(image, region)
        elif method == ReplacementMethod.BLUR:
            return self._blur_region(image, region)
        elif method == ReplacementMethod.PIXELATE:
            return self._pixelate_region(image, region)
        elif method == ReplacementMethod.EMOJI:
            return self._replace_with_emoji(image, region, custom_data)
        elif method == ReplacementMethod.BLACK_BOX:
            return self._black_box(image, region)
        elif method == ReplacementMethod.INPAINT:
            return self._inpaint_region(image, region)
        else:
            # Default to black box for unknown methods
            return self._black_box(image, region)
    
    def _get_region_mask(self, image: Image.Image, region: BoundingBox) -> Image.Image:
        """Create a mask for the given bounding box region."""
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        x1, y1, x2, y2 = region.to_xyxy()
        draw.rectangle([x1, y1, x2, y2], fill=255)
        return mask
    
    def _mask_text(self, image: Image.Image, region: BoundingBox) -> Image.Image:
        """Replace text with asterisks (****)."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = region.to_xyxy()
        
        # Draw white filled rectangle
        draw.rectangle([x1, y1, x2, y2], fill='white')
        
        # Draw asterisks
        mask_text = "****"
        try:
            # Try to use a reasonable font size
            font_size = min(region.height - 4, 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Center the text in bounding box
        text_bbox = draw.textbbox((0, 0), mask_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = x1 + (region.width - text_width) // 2
        text_y = y1 + (region.height - text_height) // 2
        
        draw.text((text_x, text_y), mask_text, fill='black', font=font)
        
        return result
    
    def _replace_with_synthetic_text(
        self, 
        image: Image.Image, 
        region: BoundingBox,
        custom_text: Optional[str] = None
    ) -> Image.Image:
        """Replace text with custom synthetic text."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = region.to_xyxy()
        
        # Draw white background
        draw.rectangle([x1, y1, x2, y2], fill='white')
        
        # Use custom text or default
        text = custom_text if custom_text else "[REDACTED]"
        
        try:
            font_size = min(region.height - 4, 16)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = x1 + (region.width - text_width) // 2
        text_y = y1 + (region.height - text_height) // 2
        
        draw.text((text_x, text_y), text, fill='gray', font=font)
        
        return result
    
    def _redact_region(self, image: Image.Image, region: BoundingBox) -> Image.Image:
        """Draw a solid black box over the region."""
        return self._black_box(image, region)
    
    def _black_box(self, image: Image.Image, region: BoundingBox) -> Image.Image:
        """Draw a black rectangle over the region."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        x1, y1, x2, y2 = region.to_xyxy()
        draw.rectangle([x1, y1, x2, y2], fill='black')
        return result
    
    def _blur_region(
        self, 
        image: Image.Image, 
        region: BoundingBox,
        blur_radius: int = 15
    ) -> Image.Image:
        """Apply Gaussian blur to the region."""
        result = image.copy()
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = region.to_xyxy()
        
        # Extract and blur region
        region_crop = result.crop((x1, y1, x2, y2))
        blurred = region_crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Paste back
        result.paste(blurred, (x1, y1))
        
        return result
    
    def _pixelate_region(
        self, 
        image: Image.Image, 
        region: BoundingBox,
        pixel_size: int = 10
    ) -> Image.Image:
        """Apply pixelation effect to the region."""
        result = image.copy()
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = region.to_xyxy()
        
        # Extract region
        region_crop = result.crop((x1, y1, x2, y2))
        
        # Resize down and up to create pixelation effect
        small_size = (max(1, region.width // pixel_size), max(1, region.height // pixel_size))
        pixelated = region_crop.resize(small_size, Image.NEAREST)
        pixelated = pixelated.resize((region.width, region.height), Image.NEAREST)
        
        # Paste back
        result.paste(pixelated, (x1, y1))
        
        return result
    
    def _replace_with_emoji(
        self,
        image: Image.Image,
        region: BoundingBox,
        emoji_type: Optional[str] = None
    ) -> Image.Image:
        """Replace region with an emoji."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = region.to_xyxy()
        
        # Draw light background
        draw.rectangle([x1, y1, x2, y2], fill='#f0f0f0')
        
        # Get emoji
        emoji = self.emoji_map.get(emoji_type, self.emoji_map['default'])
        
        # Try to draw emoji with large font
        try:
            font_size = min(region.width, region.height) - 10
            font = ImageFont.truetype("seguiemj.ttf", font_size)  # Windows emoji font
        except:
            try:
                font = ImageFont.truetype("Apple Color Emoji.ttc", font_size)  # macOS
            except:
                # Fallback: just draw a circle
                center_x = x1 + region.width // 2
                center_y = y1 + region.height // 2
                radius = min(region.width, region.height) // 3
                draw.ellipse(
                    [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                    fill='yellow',
                    outline='black',
                    width=2
                )
                return result
        
        # Draw emoji centered
        text_bbox = draw.textbbox((0, 0), emoji, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = x1 + (region.width - text_width) // 2
        text_y = y1 + (region.height - text_height) // 2
        
        draw.text((text_x, text_y), emoji, font=font, embedded_color=True)
        
        return result
    
    def _inpaint_region(self, image: Image.Image, region: BoundingBox) -> Image.Image:
        """
        Apply AI-based inpainting to the region.
        
        For production use, integrate with models like:
        - LaMa (Large Mask Inpainting)
        - Stable Diffusion Inpainting
        - OpenCV inpainting
        
        This implementation uses OpenCV's inpainting as a placeholder.
        """
        # Convert to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Create mask from bounding box
        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = region.to_xyxy()
        mask[y1:y2, x1:x2] = 255
        
        # Apply inpainting (using Telea algorithm)
        inpainted = cv2.inpaint(img_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Convert back to PIL
        inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(inpainted_rgb)
        
        return result
    
    def create_preview_with_boxes(
        self,
        image: Image.Image,
        pii_detections: List[PIIDetection],
        face_detections: List[FaceDetection],
        show_labels: bool = True
    ) -> Image.Image:
        """
        Create a preview image with polygons/boxes drawn around detections.
        
        Useful for showing users what was detected before anonymization.
        
        Args:
            image: Original image
            pii_detections: List of PII detections
            face_detections: List of face detections
            show_labels: Whether to show labels on boxes
            
        Returns:
            Preview image with regions drawn
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw PII regions in red
        for pii in pii_detections:
            bbox = pii.bbox
            x1, y1, x2, y2 = bbox.to_xyxy()
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            if show_labels:
                label = f"{pii.pii_type.value} ({pii.confidence:.2f})"
                draw.text((x1, y1 - 15), label, fill='red', font=font)
        
        # Draw face regions in blue
        for face in face_detections:
            bbox = face.bbox
            x1, y1, x2, y2 = bbox.to_xyxy()
            draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
            
            if show_labels:
                label = f"Face ({face.confidence:.2f})"
                draw.text((x1, y1 - 15), label, fill='blue', font=font)
        
        return result
    
    def anonymize_detections(
        self,
        image: Image.Image,
        pii_detections: List[PIIDetection],
        face_detections: List[FaceDetection],
        face_method: Optional[ReplacementMethod] = None,
        text_method: Optional[ReplacementMethod] = None,
        custom_data: Optional[str] = None
    ) -> Image.Image:
        """Convenience method to anonymize all detections with specified methods.
        
        Args:
            image: Original image
            pii_detections: List of PII detections to anonymize
            face_detections: List of face detections to anonymize
            face_method: Method for faces (default: BLUR, can use INPAINT for generative)
            text_method: Method for text PII (default: MASK)
            custom_data: Optional custom data for certain methods (e.g., emoji type)
            
        Returns:
            Anonymized image
        """
        # Use default methods if not specified
        face_method = face_method or self.default_face_method
        text_method = text_method or self.default_text_method
        
        # Build replacements list
        replacements = []
        
        # Add PII replacements
        for pii in pii_detections:
            replacements.append((pii.polygon, text_method, custom_data))
        
        # Add face replacements
        for face in face_detections:
            replacements.append((face.polygon, face_method, custom_data))
        
        # Apply all replacements
        return self.anonymize_image(image, replacements)
