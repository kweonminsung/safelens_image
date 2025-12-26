"""
Image generator module using Gemini and Imagen.
"""
import io
import os
import base64
from typing import Optional
from PIL import Image, ImageDraw
from google import genai
from .models import BoundingBox

class ImageGenerator:
    """Generates image patches using Gemini (for description) and Imagen (for generation)."""
    
    def __init__(
        self, 
        api_key: str = None, 
        generation_model: str = "gemini-3-flash-preview",
        imagen_model: str = "imagen-3.0-generate-002"
    ):
        """
        Initialize Image Generator.
        
        Args:
            api_key: Google AI API key
            generation_model: Gemini model to use for describing context
            imagen_model: Imagen model to use for image generation
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=api_key)
        self.generation_model = generation_model
        self.imagen_model = imagen_model

    def generate_replacement(self, image: Image.Image, region: BoundingBox) -> Optional[Image.Image]:
        """
        Generate a replacement patch for the given region using Gemini/Imagen.
        """
        try:
            # 1. Get context (crop slightly larger than bbox)
            x1, y1, x2, y2 = region.to_xyxy()
            width, height = image.size
            
            # Add padding for context
            pad_x = int(region.width * 0.5)
            pad_y = int(region.height * 0.5)
            
            ctx_x1 = max(0, x1 - pad_x)
            ctx_y1 = max(0, y1 - pad_y)
            ctx_x2 = min(width, x2 + pad_x)
            ctx_y2 = min(height, y2 + pad_y)
            
            context_img = image.crop((ctx_x1, ctx_y1, ctx_x2, ctx_y2))
            
            # 2. Ask Gemini to describe the background/texture
            # We mask the center (target) in the context image so Gemini describes the surroundings
            mask_img = context_img.copy()
            draw = ImageDraw.Draw(mask_img)
            # Calculate relative coordinates of the hole
            rel_x1 = x1 - ctx_x1
            rel_y1 = y1 - ctx_y1
            rel_x2 = x2 - ctx_x1
            rel_y2 = y2 - ctx_y1
            draw.rectangle([rel_x1, rel_y1, rel_x2, rel_y2], fill='black')
            
            prompt = "Describe the background texture, color, and pattern of this image, ignoring the black masked rectangle in the center. Keep it concise, e.g., 'white concrete wall', 'blue denim fabric', 'human skin'."
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            mask_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            response = self.client.models.generate_content(
                model=self.generation_model,
                contents=[
                    prompt,
                    {"inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(img_byte_arr.getvalue()).decode()
                    }}
                ]
            )
            description = response.text.strip()
            print(f"[INFO] Gemini description for generation: {description}")
            
            # 3. Generate texture using Imagen (if available)
            try:
                # Note: This requires the API key to have access to Imagen models
                imagen_response = self.client.models.generate_images(
                    model=self.imagen_model,
                    prompt=f"Texture of {description}. High quality, seamless pattern.",
                    config={"number_of_images": 1, "aspect_ratio": "1:1"}
                )
                if imagen_response.generated_images:
                    gen_img_data = imagen_response.generated_images[0].image.image_bytes
                    gen_img = Image.open(io.BytesIO(gen_img_data))
                    return gen_img
            except Exception as e:
                print(f"[WARNING] Imagen generation failed: {e}. Falling back to None.")
                return None

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return None
