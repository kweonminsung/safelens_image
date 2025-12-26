"""
Gemini-based unified detector for PII and faces.
Uses Google Gemini Vision API for detection.
"""
import uuid
import os
import json
from typing import List
from PIL import Image, ImageOps
import io
import base64
from google import genai

from .models import (
    PIIDetection,
    FaceDetection,
    BoundingBox,
    PIIType,
    DetectionType,
)


# System prompt for Gemini detection
DETECTION_PROMPT_TEMPLATE = """You are an AI system performing privacy-safe image redaction and replacement.

PHASE 1 — DETECTION (NO MODIFICATION)

Analyze the provided image and DETECT ONLY the following:

1. **Text containing personal information (PII)** - ONLY detect if text is CLEARLY VISIBLE and LEGIBLE:
   - Phone numbers (with digits visible)
   - Street addresses (with numbers and street names)
   - Personal names (when clearly readable)
   - Email addresses (with @ symbol visible)
   - Vehicle license plates / car number plates (with plate number visible)
   
   DO NOT detect:
   - Blurry or illegible text
   - Logos, brand names, or store signs
   - General labels or descriptions
   - Random patterns that look like text

2. **Human faces** - ONLY detect clear, recognizable human faces:
   - Must show eyes, nose, and mouth
   - Must be a real human face (not drawings, logos, or objects)
   
   DO NOT detect:
   - Partial faces or side profiles without clear features
   - Objects that vaguely resemble faces
   - Mannequins or statues unless very realistic

For EACH detected item, output a JSON object with:
- id: unique string identifier
- type: "text_pii" or "face"
- label: for text_pii: "phone" | "address" | "name" | "email" | "license_plate" ; for face: "face"
- bbox: bounding box as [x_min, y_min, x_max, y_max] where:
  * VALUES MUST BE INTEGERS BETWEEN 0 AND 1000 (NORMALIZED COORDINATES)
  * 0 represents top/left edge, 1000 represents bottom/right edge
  * Example: [0, 0, 1000, 1000] is the full image
  * [x_min, y_min, x_max, y_max] order
- confidence: number between 0 and 1

CRITICAL COORDINATE RULES:
- Coordinates are NORMALIZED to 0-1000 scale
- x values range from 0 to 1000
- y values range from 0 to 1000
- Format: [x_min, y_min, x_max, y_max] where x_min < x_max and y_min < y_max
- DO NOT use pixel values, use 0-1000 scale

IMPORTANT:
- Detection ONLY in this phase
- Do NOT anonymize, blur, or modify anything
- Do NOT guess identities
- Face recognition or identity inference is NOT allowed
- Return ONLY valid JSON array of detections

Output format:
[
  {{
    "id": "uuid-1",
    "type": "text_pii",
    "label": "phone",
    "bbox": [100, 200, 300, 250],
    "confidence": 0.95
  }},
  {{
    "id": "uuid-2",
    "type": "face",
    "label": "face",
    "bbox": [500, 600, 800, 900],
    "confidence": 0.98
  }}
]

Return ONLY the JSON array, no additional text."""


class GeminiDetector:
    """Unified detector using Gemini Vision API."""
    
    def __init__(
        self, 
        api_key: str = None, 
        model_name: str = "gemini-2.5-flash-image",
        min_confidence: float = 0.7
    ):
        """
        Initialize Gemini detector.
        
        Args:
            api_key: Google AI API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use
            min_confidence: Minimum confidence threshold for detections (0.0-1.0)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.min_confidence = min_confidence
    
    def detect(self, image: Image.Image) -> tuple[List[PIIDetection], List[FaceDetection]]:
        """
        Detect PII and faces in an image using Gemini.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (pii_detections, face_detections)
        """

        
        # Get original image dimensions
        orig_width, orig_height = image.size
        print(f"\n[INFO] Processing image: {orig_width}x{orig_height} pixels")
        
        # Resize image if too large (Gemini works better with standard sizes)
        # Max dimension 1024px is usually a good balance for accuracy/speed
        MAX_DIMENSION = 1024
        scale_factor = 1.0
        
        if orig_width > MAX_DIMENSION or orig_height > MAX_DIMENSION:
            if orig_width > orig_height:
                scale_factor = MAX_DIMENSION / orig_width
                new_width = MAX_DIMENSION
                new_height = int(orig_height * scale_factor)
            else:
                scale_factor = MAX_DIMENSION / orig_height
                new_height = MAX_DIMENSION
                new_width = int(orig_width * scale_factor)
                
            print(f"[INFO] Resizing image to {new_width}x{new_height} (scale: {scale_factor:.4f}) for better detection")
            processing_image = image.resize((new_width, new_height), Image.LANCZOS)
            proc_width, proc_height = new_width, new_height
        else:
            processing_image = image
            proc_width, proc_height = orig_width, orig_height
        
        # Format prompt with processing image dimensions
        detection_prompt = DETECTION_PROMPT_TEMPLATE.format(width=proc_width, height=proc_height)
        
        # Convert image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        processing_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Call Gemini API with new client
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                detection_prompt,
                {"inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(img_byte_arr.getvalue()).decode()
                }}
            ]
        )
        
        # Parse response
        try:
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            detections = json.loads(response_text)
            
            if not isinstance(detections, list):
                raise ValueError("Response is not a list")
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse Gemini response: {e}")
            print(f"Response text: {response.text}")
            return [], []
        
        # Convert to our models
        pii_detections = []
        face_detections = []
        
        for det in detections:
            detection_id = det.get("id", str(uuid.uuid4()))
            detection_type = det.get("type")
            label = det.get("label")
            bbox_coords = det.get("bbox", [])
            confidence = det.get("confidence", 0.0)
            
            # Debug: Print original coordinates from Gemini
            print(f"\n[DEBUG] Gemini returned {detection_type} ({label}):")
            print(f"  Original bbox: {bbox_coords}")
            print(f"  Processing size: {proc_width} x {proc_height}")
            
            # Filter by confidence threshold
            if confidence < self.min_confidence:
                continue
            
            # Parse bbox [x_min, y_min, x_max, y_max]
            if len(bbox_coords) != 4:
                print(f"  [WARNING] Invalid bbox format, skipping")
                continue
                
            x_min, y_min, x_max, y_max = bbox_coords
            
            # Handle 0-1000 normalized coordinates (preferred)
            if all(0 <= c <= 1000 for c in bbox_coords) and any(c > 1 for c in bbox_coords):
                print(f"  [INFO] Detected 0-1000 normalized coords, converting to pixels")
                x_min = int((x_min / 1000.0) * orig_width)
                x_max = int((x_max / 1000.0) * orig_width)
                y_min = int((y_min / 1000.0) * orig_height)
                y_max = int((y_max / 1000.0) * orig_height)
            
            # Handle 0-1 normalized coordinates (fallback)
            elif all(0 <= c <= 1.0 for c in bbox_coords):
                print(f"  [INFO] Detected 0-1 normalized coords, converting to pixels")
                x_min = int(x_min * orig_width)
                x_max = int(x_max * orig_width)
                y_min = int(y_min * orig_height)
                y_max = int(y_max * orig_height)
                
            # Handle absolute pixel coordinates (fallback)
            else:
                print(f"  [INFO] Detected absolute pixel coords")
                # If we resized, we need to scale back? 
                # But if model used proc_width/height, we need to know.
                # Assuming model followed instructions and used 0-1000, this block shouldn't be hit often.
                # If it returns pixels relative to resized image:
                if scale_factor != 1.0 and x_max <= proc_width and y_max <= proc_height:
                     print(f"  [INFO] Scaling up from processing size")
                     x_min = int(x_min / scale_factor)
                     x_max = int(x_max / scale_factor)
                     y_min = int(y_min / scale_factor)
                     y_max = int(y_max / scale_factor)
            
            # Validate bbox
            if x_min >= x_max or y_min >= y_max:
                print(f"  [WARNING] Invalid bbox: x_min >= x_max or y_min >= y_max, skipping")
                continue
            
            # Clamp coordinates to original image bounds
            x_min = max(0, min(x_min, orig_width))
            x_max = max(0, min(x_max, orig_width))
            y_min = max(0, min(y_min, orig_height))
            y_max = max(0, min(y_max, orig_height))
            
            # Ensure bbox still valid after clamping
            if x_min >= x_max or y_min >= y_max:
                print(f"  [WARNING] bbox became invalid after clamping, skipping")
                continue
            
            print(f"  Final pixel bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            print(f"  Bbox size: {x_max - x_min} x {y_max - y_min} pixels")
            
            # Create BoundingBox object
            bbox = BoundingBox(
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min
            )
            
            if detection_type == "text_pii":
                # Map label to PIIType
                pii_type_map = {
                    "phone": PIIType.PHONE,
                    "email": PIIType.EMAIL,
                    "address": PIIType.ADDRESS,
                    "name": PIIType.NAME,
                    "license_plate": PIIType.LICENSE_PLATE,
                }
                pii_type = pii_type_map.get(label, PIIType.OTHER)
                
                pii_det = PIIDetection(
                    detection_id=detection_id,
                    detection_type=DetectionType.TEXT_PII,
                    pii_type=pii_type,
                    text=det.get("text", ""),  # Gemini might provide text
                    bbox=bbox,
                    confidence=confidence
                )
                pii_detections.append(pii_det)
                
            elif detection_type == "face":
                face_det = FaceDetection(
                    detection_id=detection_id,
                    detection_type=DetectionType.FACE,
                    bbox=bbox,
                    confidence=confidence
                )
                face_detections.append(face_det)
        
        return pii_detections, face_detections
