"""
Image generator module using Gemini and Imagen.
"""
import io
import os
import uuid
import base64
from typing import Optional
from PIL import Image, ImageDraw
from google import genai
from .models import BoundingBox, PIIType, PII_REPLACEMENT_VALUES

class ImageGenerator:
    """Generates image patches using Imagen Inpainting."""
    
    def __init__(
        self, 
        api_key: str = None, 
        # imagen_model: str = "gemini-3-pro-image-preview"
        imagen_model: str = "gemini-2.5-flash-image",
        mask_padding: int = 10
    ):
        """
        Initialize Image Generator.
        
        Args:
            api_key: Google AI API key
            imagen_model: Model to use for image generation (default: gemini-3-pro-image-preview)
            mask_padding: Padding (in pixels) to add around the bbox for better context (default: 10)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=api_key)
        self.imagen_model = imagen_model
        self.mask_padding = mask_padding

    def generate_replacement(self, image: Image.Image, region: BoundingBox, label: str = None) -> Optional[Image.Image]:
        """
        Generate a replacement patch for the given region using Gemini.
        
        Args:
            image: Original image
            region: Bounding box of the region to replace
            label: Label of the object being replaced (PIIType value or "face")
        """
        try:
            # 1. Create mask for the region (white = area to edit)
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)
            x1, y1, x2, y2 = region.to_xyxy()
            
            # Add padding to mask for better context (but keep original bbox for final crop)
            mask_x1 = max(0, x1 - self.mask_padding)
            mask_y1 = max(0, y1 - self.mask_padding)
            mask_x2 = min(image.width, x2 + self.mask_padding)
            mask_y2 = min(image.height, y2 + self.mask_padding)
            
            draw.rectangle([mask_x1, mask_y1, mask_x2, mask_y2], fill=255)
            
            print(f"[DEBUG] Original bbox: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"[DEBUG] Padded mask: ({mask_x1}, {mask_y1}) to ({mask_x2}, {mask_y2}) [padding: {self.mask_padding}px]")
            
            # Debug: Save mask image
            os.makedirs("temp", exist_ok=True)
            mask_uuid = uuid.uuid4()
            mask_path = f"temp/debug_mask_{mask_uuid}.png"
            mask.save(mask_path)
            print(f"[DEBUG] Saved mask to {mask_path}")
            
            # 2. Call Gemini Generate Content
            try:
                from google.genai import types
                import io
                
                prompt_text = (
                    "You are an expert image editor. Your task is to fill in the masked area (indicated by the white region in the second image) "
                    "of the first image seamlessly. The filled area should match the surrounding background texture, lighting, and context perfectly. "
                    "Do not change any other part of the image. Output only the modified image."
                )
                
                if label:
                    # Get replacement value from predefined constants
                    replacement_value = None
                    try:
                        # Try to get PIIType from label string
                        pii_type = PIIType(label)
                        replacement_value = PII_REPLACEMENT_VALUES.get(pii_type)
                    except (ValueError, KeyError):
                        # Not a valid PIIType, might be "face" or other
                        pass
                    
                    label_lower = label.lower()
                    
                    if replacement_value:
                        # Use predefined replacement value
                        if "license" in label_lower or "plate" in label_lower:
                            prompt_text += f" The masked area contained a license plate. Replace it with a realistic license plate showing '{replacement_value}', maintaining the same style, format, color, and appearance. The new plate must look natural and realistic."
                        elif "phone" in label_lower or "telephone" in label_lower or "mobile" in label_lower:
                            prompt_text += f" The masked area contained a phone number. Replace it with the phone number '{replacement_value}' in the same format, style, and font."
                        elif "id" in label_lower or "card" in label_lower:
                            prompt_text += f" The masked area contained sensitive identification information ({label}). Replace it with '{replacement_value}' in the same format and style."
                        elif "address" in label_lower or "postal" in label_lower:
                            prompt_text += f" The masked area contained address information. Replace it with '{replacement_value}' in the same format and style."
                        elif "email" in label_lower:
                            prompt_text += f" The masked area contained an email address. Replace it with '{replacement_value}' in the same format and style."
                        elif "name" in label_lower:
                            prompt_text += f" The masked area contained a name. Replace it with '{replacement_value}' in the same format and style."
                        elif "birth" in label_lower or "date" in label_lower:
                            prompt_text += f" The masked area contained a date. Replace it with '{replacement_value}' in the same format and style."
                        else:
                            prompt_text += f" The masked area contained '{label}'. Replace it with '{replacement_value}' maintaining the same style and appearance."
                    else:
                        # No predefined value, use random or background fill
                        if "license" in label_lower or "plate" in label_lower:
                            prompt_text += " The masked area contained a license plate. Replace it with a realistic license plate with DIFFERENT RANDOM numbers and letters, maintaining the same style, format, color, and appearance. The new plate must look natural and realistic."
                        elif "phone" in label_lower or "telephone" in label_lower or "mobile" in label_lower:
                            prompt_text += " The masked area contained a phone number. Replace it with a DIFFERENT RANDOM phone number in the same format, style, and font."
                        elif "face" in label_lower:
                            prompt_text += " The masked area contained a face. Please blur or obscure it naturally while maintaining the overall image composition."
                        else:
                            prompt_text += f" The masked area contained a {label}. Please replace it with a natural background that fits the context perfectly."
                
                # Add coordinates to prompt
                prompt_text += f" The coordinates of the area to be modified are: ({x1}, {y1}) to ({x2}, {y2})."

                print(f"[DEBUG] Prompt: {prompt_text}")

                response = self.client.models.generate_content(
                    model=self.imagen_model,
                    contents=[
                        prompt_text,
                        image,
                        mask
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"]
                    )
                )
                
                if response.parts:
                    for part in response.parts:
                        if part.inline_data:
                            gen_img = Image.open(io.BytesIO(part.inline_data.data))
                            
                            # Debug: Save generated image
                            os.makedirs("temp", exist_ok=True)
                            debug_path = f"temp/debug_gen_{uuid.uuid4()}.png"
                            gen_img.save(debug_path)
                            print(f"[DEBUG] Saved generated image to {debug_path}")
                            print(f"[DEBUG] Original size: {image.size}, Generated size: {gen_img.size}")
                            
                            # Check if generated image size matches original
                            if gen_img.size != image.size:
                                print(f"[WARNING] Size mismatch! Resizing {gen_img.size} -> {image.size}")
                                gen_img = gen_img.resize(image.size, Image.Resampling.LANCZOS)

                            # Crop the patch from the generated full image
                            patch = gen_img.crop((x1, y1, x2, y2))
                            print(f"[DEBUG] Patch size: {patch.size}, Expected: ({x2-x1}, {y2-y1})")
                            return patch
                
                print("[WARNING] No image part found in response.")
                return None
                    
            except Exception as e:
                print(f"[WARNING] Gemini generation failed: {e}")
                return None

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return None


