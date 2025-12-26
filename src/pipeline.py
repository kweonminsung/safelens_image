"""
Pipeline orchestrator that coordinates detection and anonymization.
"""
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

from .models import (
    DetectionResult,
    PIIDetection,
    FaceDetection,
    AnonymizationRequest,
    AnonymizationResult,
    ReplacementRequest,
    ReplacementMethod,
)
from .gemini_detector import GeminiDetector
from .image_generator import ImageGenerator
from .anonymizer import Anonymizer


class PrivacyPipeline:
    """
    Main pipeline for image privacy sanitization.
    
    This orchestrates the entire workflow:
    1. Image upload
    2. Detection (PII text + faces) using Gemini Vision API
    3. User selection
    4. Anonymization
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        min_confidence: float = 0.7,
        detection_model: str = "gemini-3-flash-preview",
        generation_model: str = "gemini-1.5-flash",
        imagen_model: str = "imagen-3.0-generate-002",
        default_face_method: ReplacementMethod = ReplacementMethod.BLUR,
        default_text_method: ReplacementMethod = ReplacementMethod.BLUR,
        upload_dir: str = "uploads",
        output_dir: str = "outputs"
    ):
        """
        Initialize the pipeline.
        
        Args:
            gemini_api_key: Google Gemini API key (if None, will read from env)
            min_confidence: Minimum confidence for detections
            detection_model: Gemini model to use for detection
            generation_model: Gemini model to use for describing context for generation
            imagen_model: Imagen model to use for image generation
            default_face_method: Default anonymization for faces (BLUR or INPAINT)
            default_text_method: Default anonymization for text PII (MASK or REDACT)
            upload_dir: Directory to save uploaded images
            output_dir: Directory to save anonymized images
        """
        # Initialize Gemini detector
        self.detector = GeminiDetector(
            api_key=gemini_api_key,
            min_confidence=min_confidence,
            detection_model=detection_model
        )
        
        # Initialize Image Generator
        self.generator = ImageGenerator(
            api_key=gemini_api_key,
            generation_model=generation_model,
            imagen_model=imagen_model
        )
        
        self.anonymizer = Anonymizer(
            generator=self.generator,
            default_face_method=default_face_method,
            default_text_method=default_text_method
        )
        
        # Setup storage directories
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # In-memory storage for images (for quick access)
        self._image_cache: Dict[str, Image.Image] = {}
        self._detection_cache: Dict[str, DetectionResult] = {}
    
    def detect(self, image: Image.Image, image_id: Optional[str] = None) -> DetectionResult:
        """
        Run detection pipeline on an image using Gemini Vision API.
        
        Args:
            image: PIL Image object
            image_id: Optional ID for the image (generated if not provided)
            
        Returns:
            DetectionResult with all detected PII and faces
        """
        if image_id is None:
            image_id = str(uuid.uuid4())
        
        # Store image in memory for quick access
        self._image_cache[image_id] = image.copy()
        
        # Save original image to disk
        image_path = self.upload_dir / f"{image_id}.png"
        image.save(image_path, "PNG")
        
        # Run Gemini detection (unified PII + face detection)
        pii_detections, face_detections = self.detector.detect(image)
        
        # Create result
        result = DetectionResult(
            image_id=image_id,
            pii_detections=pii_detections,
            face_detections=face_detections,
            image_width=image.width,
            image_height=image.height
        )
        
        # Cache the detection result
        self._detection_cache[image_id] = result
        
        return result
    
    def anonymize(self, request: AnonymizationRequest) -> tuple[Image.Image, AnonymizationResult]:
        """
        Apply anonymization to selected detections.
        
        Args:
            request: AnonymizationRequest with selected replacements
            
        Returns:
            Tuple of (anonymized_image, result_info)
        """
        # Retrieve cached image and detections
        image = self._image_cache.get(request.image_id)
        detections = self._detection_cache.get(request.image_id)
        
        if image is None:
            raise ValueError(f"Image not found: {request.image_id}")
        
        if detections is None:
            raise ValueError(f"Detections not found for image: {request.image_id}")
        
        # Build a map of detection_id -> detection
        detection_map = {}
        for pii in detections.pii_detections:
            detection_map[pii.detection_id] = pii
        for face in detections.face_detections:
            detection_map[face.detection_id] = face
        
        # Prepare replacements for anonymizer
        replacements = []
        applied_ids = []
        
        for replacement_req in request.replacements:
            detection = detection_map.get(replacement_req.detection_id)
            if detection is None:
                continue
            
            bbox = detection.bbox
            method = replacement_req.method
            custom_data = replacement_req.custom_text
            
            replacements.append((bbox, method, custom_data))
            applied_ids.append(replacement_req.detection_id)
        
        # Apply anonymization
        anonymized_image = self.anonymizer.anonymize_image(image, replacements)
        
        # Save anonymized image to disk
        output_path = self.output_dir / f"{request.image_id}_anonymized.png"
        anonymized_image.save(output_path, "PNG")
        
        # Create result
        result = AnonymizationResult(
            image_id=request.image_id,
            applied_replacements=applied_ids,
            output_format=request.output_format,
            message=f"Successfully anonymized {len(applied_ids)} regions. Saved to {output_path}"
        )
        
        return anonymized_image, result
    
    def create_preview(self, image_id: str, show_labels: bool = True) -> Optional[Image.Image]:
        """
        Create a preview image with all detections highlighted.
        
        Args:
            image_id: ID of the image
            show_labels: Whether to show labels on bounding boxes
            
        Returns:
            Preview image or None if not found
        """
        image = self._image_cache.get(image_id)
        detections = self._detection_cache.get(image_id)
        
        if image is None or detections is None:
            return None
        
        return self.anonymizer.create_preview_with_boxes(
            image,
            detections.pii_detections,
            detections.face_detections,
            show_labels=show_labels
        )
    
    def clear_cache(self, image_id: Optional[str] = None):
        """
        Clear cached images and detections (also deletes files from disk).
        
        Args:
            image_id: Specific image to clear, or None to clear all
        """
        if image_id:
            self._image_cache.pop(image_id, None)
            self._detection_cache.pop(image_id, None)
            
            # Delete files from disk
            upload_path = self.upload_dir / f"{image_id}.png"
            output_path = self.output_dir / f"{image_id}_anonymized.png"
            
            if upload_path.exists():
                upload_path.unlink()
            if output_path.exists():
                output_path.unlink()
        else:
            self._image_cache.clear()
            self._detection_cache.clear()
            
            # Delete all files from disk
            for file in self.upload_dir.glob("*.png"):
                file.unlink()
            for file in self.output_dir.glob("*.png"):
                file.unlink()
