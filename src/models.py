"""
Data models for image privacy sanitization pipeline.
All coordinates are in pixel space.
Supports both polygon and bounding box representations.
"""
from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class PIIType(str, Enum):
    """Types of personally identifiable information."""
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    NAME = "name"
    LICENSE_PLATE = "license_plate"
    ID_NUMBER = "id_number"
    CREDIT_CARD = "credit_card"
    DATE_OF_BIRTH = "date_of_birth"
    OTHER = "other"


class DetectionType(str, Enum):
    """Types of detections in an image."""
    TEXT_PII = "text_pii"
    FACE = "face"


class ReplacementMethod(str, Enum):
    """Methods for replacing detected regions."""
    # Text replacements
    MASK = "mask"  # Replace with ***
    SYNTHETIC_TEXT = "synthetic_text"  # Generate synthetic text
    REDACT = "redact"  # Black box
    
    # Face replacements
    BLUR = "blur"  # Gaussian blur
    PIXELATE = "pixelate"  # Pixelation
    EMOJI = "emoji"  # Replace with emoji
    INPAINT = "inpaint"  # AI-based inpainting
    BLACK_BOX = "black_box"  # Black rectangle


class Polygon(BaseModel):
    """Polygon region in pixel coordinates."""
    points: List[Tuple[int, int]] = Field(..., description="List of (x, y) coordinates defining the polygon boundary")
    
    def to_bbox(self) -> 'BoundingBox':
        """Convert polygon to bounding box (for compatibility)."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min
        )
    
    def to_tuple_list(self) -> List[Tuple[int, int]]:
        """Get list of (x, y) tuples."""
        return self.points


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates."""
    x: int = Field(..., description="Top-left x coordinate in pixels")
    y: int = Field(..., description="Top-left y coordinate in pixels")
    width: int = Field(..., ge=1, description="Width in pixels")
    height: int = Field(..., ge=1, description="Height in pixels")
    
    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_xywh(self) -> tuple[int, int, int, int]:
        """Convert to (x, y, w, h) format."""
        return (self.x, self.y, self.width, self.height)
    
    def to_polygon(self) -> Polygon:
        """Convert bounding box to polygon (4 corners)."""
        x1, y1, x2, y2 = self.to_xyxy()
        return Polygon(points=[
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2)   # Bottom-left
        ])


class TextDetection(BaseModel):
    """Detected text with OCR results."""
    text: str = Field(..., description="Extracted text content")
    bbox: BoundingBox = Field(..., description="Bounding box of the text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")


class PIIDetection(BaseModel):
    """Detected PII from text."""
    detection_id: str = Field(..., description="Unique identifier for this detection")
    detection_type: DetectionType = Field(default=DetectionType.TEXT_PII)
    pii_type: PIIType = Field(..., description="Type of PII detected")
    text: str = Field(..., description="The detected PII text")
    bbox: BoundingBox = Field(..., description="Bounding box of the PII")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class FaceDetection(BaseModel):
    """Detected face (bounding box region only, no identity)."""
    detection_id: str = Field(..., description="Unique identifier for this detection")
    detection_type: DetectionType = Field(default=DetectionType.FACE)
    bbox: BoundingBox = Field(..., description="Bounding box of the face")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class DetectionResult(BaseModel):
    """Combined detection result from the pipeline."""
    image_id: str = Field(..., description="Unique identifier for the processed image")
    pii_detections: List[PIIDetection] = Field(
        default_factory=list, description="All detected PII instances"
    )
    face_detections: List[FaceDetection] = Field(
        default_factory=list, description="All detected faces"
    )
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")


class ReplacementRequest(BaseModel):
    """Request to replace specific detections."""
    detection_id: str = Field(..., description="ID of the detection to replace")
    method: ReplacementMethod = Field(..., description="Replacement method to use")
    custom_text: Optional[str] = Field(
        None, description="Custom text for synthetic_text method"
    )


class AnonymizationRequest(BaseModel):
    """Request to anonymize selected regions in an image."""
    image_id: str = Field(..., description="ID of the image to anonymize")
    replacements: List[ReplacementRequest] = Field(
        ..., description="List of replacements to apply"
    )
    output_format: str = Field(
        default="png", description="Output image format (png, jpg, webp)"
    )


class AnonymizationResult(BaseModel):
    """Result of anonymization operation."""
    image_id: str = Field(..., description="ID of the anonymized image")
    applied_replacements: List[str] = Field(
        ..., description="List of detection IDs that were replaced"
    )
    output_format: str = Field(..., description="Format of the output image")
    message: str = Field(default="Anonymization completed successfully")
