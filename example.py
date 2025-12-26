"""
Example usage script demonstrating the SaveLens pipeline with real image.
"""
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from src.pipeline import PrivacyPipeline
from src.models import AnonymizationRequest, ReplacementRequest, ReplacementMethod

# Load environment variables from .env file
load_dotenv()


def main():
    """Run the example pipeline with public/example1.jpg."""
    print("=" * 60)
    print("SaveLens Image Privacy Sanitization - Example Usage")
    print("=" * 60)
    
    # Load example image
    print("\n1. Loading example image...")
    image_path = Path("public/example1.jpg")
    if not image_path.exists():
        print(f"❌ Example image not found: {image_path}")
        print("   Please place an image at public/example1.jpg")
        return
    
    image = Image.open(image_path)
    print(f"✓ Loaded image: {image_path}")
    print(f"   Size: {image.width}x{image.height} pixels")
    
    # Initialize pipeline
    print("\n2. Initializing pipeline with Gemini Vision API...")
    pipeline = PrivacyPipeline(
        min_confidence=0.7,
        default_face_method=ReplacementMethod.BLUR,
        default_text_method=ReplacementMethod.MASK
    )
    print("✓ Pipeline initialized")
    
    # Step 1: Detection
    print("\n3. Running detection with Gemini...")
    result = pipeline.detect(image)
    print(f"✓ Detection complete!")
    print(f"   - Found {len(result.pii_detections)} PII instances")
    print(f"   - Found {len(result.face_detections)} faces")
    print(f"   - Image ID: {result.image_id}")
    print(f"   - Saved to: uploads/{result.image_id}.png")
    
    # Show detected items
    if result.pii_detections:
        print("\n4. Detected PII:")
        for i, pii in enumerate(result.pii_detections, 1):
            print(f"   {i}. Type: {pii.pii_type.value}")
            print(f"      Text: {pii.text if pii.text else '(detected)'}")
            print(f"      Confidence: {pii.confidence:.2f}")
            bbox = pii.bbox
            print(f"      BBox: x={bbox.x}, y={bbox.y}, w={bbox.width}, h={bbox.height}")
    else:
        print("\n4. No PII detected")
    
    if result.face_detections:
        print("\n   Detected Faces:")
        for i, face in enumerate(result.face_detections, 1):
            print(f"   {i}. Confidence: {face.confidence:.2f}")
            bbox = face.bbox
            print(f"      BBox: x={bbox.x}, y={bbox.y}, w={bbox.width}, h={bbox.height}")
    
    # Step 2: Create preview
    print("\n5. Creating preview with detection boxes...")
    preview = pipeline.create_preview(result.image_id, show_labels=True)
    if preview:
        preview_path = Path('outputs') / f'{result.image_id}_preview.png'
        preview.save(preview_path)
        print(f"✓ Preview saved as '{preview_path}'")
    
    # Step 3: Prepare anonymization
    print("\n6. Preparing anonymization request...")
    replacements = []
    
    # Mask all PII text
    for pii in result.pii_detections:
        replacements.append(ReplacementRequest(
            detection_id=pii.detection_id,
            detection_type="pii",
            method=ReplacementMethod.MASK
        ))
    
    # Blur all faces
    for face in result.face_detections:
        replacements.append(ReplacementRequest(
            detection_id=face.detection_id,
            detection_type="face",
            method=ReplacementMethod.BLUR
        ))
    
    print(f"✓ Prepared {len(replacements)} replacements")
    
    # Step 4: Apply anonymization
    print("\n7. Applying anonymization...")
    request = AnonymizationRequest(
        image_id=result.image_id,
        replacements=replacements,
        output_format='png'
    )
    
    anonymized_img, anon_result = pipeline.anonymize(request)
    print("✓ Anonymization complete!")
    print(f"   {anon_result.message}")
    
    print("\n" + "=" * 60)
    print("Example complete! Generated files:")
    print(f"  - uploads/{result.image_id}.png: Original")
    print(f"  - outputs/{result.image_id}_preview.png: Preview with boxes")
    print(f"  - outputs/{result.image_id}_anonymized.png: Anonymized")
    print("=" * 60)


if __name__ == "__main__":
    main()
