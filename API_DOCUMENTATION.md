# SaveLens Image Privacy Sanitization API

Privacy-safe image sanitization API with PII detection and face detection using Google Gemini Vision API.

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

**GET** `/`

Check if the API is running.

**Response:**

```json
{
  "service": "SaveLens Image Privacy Sanitization API",
  "version": "0.1.0",
  "status": "healthy"
}
```

---

### 2. Upload Image

**POST** `/upload`

Upload an image and receive a unique UUID for later processing. The system automatically generates both high-quality (original) and low-quality (thumbnail) versions.

**Request:**

- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (file, required): Image file to upload (PNG, JPG, JPEG, WEBP)

**Response:**

```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Image uploaded successfully"
}
```

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@image.jpg"
```

---

### 3. Detect PII and Faces

**POST** `/detect/{image_id}`

Run detection on an uploaded image to find PII (personally identifiable information) and faces.

**Path Parameters:**

- `image_id` (string, required): UUID returned from upload endpoint

**Response:**

```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_width": 1920,
  "image_height": 1080,
  "pii_detections": [
    {
      "detection_id": "pii-001",
      "detection_type": "text_pii",
      "pii_type": "phone",
      "text": "010-1234-5678",
      "bbox": {
        "x": 100,
        "y": 200,
        "width": 150,
        "height": 30
      },
      "confidence": 0.95
    }
  ],
  "face_detections": [
    {
      "detection_id": "face-001",
      "detection_type": "face",
      "bbox": {
        "x": 500,
        "y": 300,
        "width": 200,
        "height": 250
      },
      "confidence": 0.98
    }
  ]
}
```

**PII Types:**

- `phone`: Phone numbers
- `email`: Email addresses
- `address`: Street addresses
- `name`: Personal names
- `license_plate`: Vehicle license plates
- `id_number`: ID/SSN numbers
- `credit_card`: Credit card numbers
- `date_of_birth`: Birth dates
- `qrcode`: QR codes
- `barcode`: Barcodes
- `signboard`: Store signs and signboards
- `other`: Other PII

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/detect/550e8400-e29b-41d4-a716-446655440000"
```

---

### 4. Anonymize Image

**POST** `/anonymize`

Apply anonymization to specific regions of the image using bounding boxes with PII type labels. The system automatically generates both high-quality (original resolution) and low-quality (thumbnail) versions of the anonymized image.

**Request Body:**

```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "regions": [
    {
      "bbox": {
        "x": 100,
        "y": 200,
        "width": 150,
        "height": 30
      },
      "pii_type": "phone"
    },
    {
      "bbox": {
        "x": 500,
        "y": 300,
        "width": 200,
        "height": 250
      },
      "pii_type": "face"
    }
  ],
  "method": "generate"
}
```

**Parameters:**

- `image_id` (string, required): UUID of the image
- `regions` (array, required): List of regions to anonymize
  - `bbox` (object): Bounding box coordinates
    - `x` (integer): X coordinate (top-left)
    - `y` (integer): Y coordinate (top-left)
    - `width` (integer): Width of the region
    - `height` (integer): Height of the region
  - `pii_type` (string, optional): PII type or "face" (e.g., "phone", "email", "qrcode", "barcode", "face")
- `method` (string, optional): Anonymization method (default: "generate")
  - `generate`: AI-powered generative fill using Gemini
  - `blur`: Gaussian blur
  - `black_box`: Black rectangle overlay
  - `mosaic`: Pixelated mosaic effect

**Response:**

```json
{
  "original_image_id": "550e8400-e29b-41d4-a716-446655440000",
  "anonymized_image_id": "660e8400-e29b-41d4-a716-446655440001",
  "success": true,
  "message": "Successfully anonymized 2 regions",
  "processed_count": 2
}
```

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/anonymize" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "550e8400-e29b-41d4-a716-446655440000",
    "regions": [
      {
        "bbox": {"x": 100, "y": 200, "width": 150, "height": 30},
        "pii_type": "phone"
      }
    ],
    "method": "generate"
  }'
```

---

### 5. Download Image

**GET** `/download/{image_id}`

Download an image by UUID (original or anonymized) with quality option.

**Path Parameters:**

- `image_id` (string, required): UUID of the image (use `original_image_id` or `anonymized_image_id` from anonymize response)

**Query Parameters:**

- `quality` (string, optional): Image quality - `high` (original resolution) or `low` (thumbnail) (default: "high")
- `format` (string, optional): Output format - `png`, `jpg`, `jpeg`, `webp` (default: "png")

**Response:**

- Binary image file with appropriate Content-Type

**Examples (curl):**

```bash
# Download original high-quality image
curl -X GET "http://localhost:8000/download/550e8400-e29b-41d4-a716-446655440000?quality=high" \
  -o original_high.png

# Download thumbnail
curl -X GET "http://localhost:8000/download/550e8400-e29b-41d4-a716-446655440000?quality=low" \
  -o original_thumb.png

# Download anonymized image as JPEG
curl -X GET "http://localhost:8000/download/660e8400-e29b-41d4-a716-446655440001?quality=high&format=jpg" \
  -o anonymized.jpg
```

---

### 6. Clear Image Cache

**DELETE** `/clear/{image_id}`

Remove a specific image and its detection data from cache.

**Path Parameters:**

- `image_id` (string, required): UUID of the image to clear

**Response:**

```json
{
  "message": "Cache cleared for image 550e8400-e29b-41d4-a716-446655440000"
}
```

**Example (curl):**

```bash
curl -X DELETE "http://localhost:8000/clear/550e8400-e29b-41d4-a716-446655440000"
```

---

### 7. Clear All Cache

**DELETE** `/clear`

Remove all cached images and detection data.

**Response:**

```json
{
  "message": "All cache cleared"
}
```

**Example (curl):**

```bash
curl -X DELETE "http://localhost:8000/clear"
```

---

## Complete Workflow Example

### Step 1: Upload Image

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@photo.jpg"
```

Response:

```json
{
  "image_id": "abc-123",
  "message": "Image uploaded successfully"
}
```

### Step 2: Detect PII and Faces

```bash
curl -X POST "http://localhost:8000/detect/abc-123"
```

Response:

```json
{
  "image_id": "abc-123",
  "pii_detections": [
    {
      "detection_id": "pii-1",
      "pii_type": "phone",
      "bbox": { "x": 100, "y": 200, "width": 150, "height": 30 },
      "confidence": 0.95
    }
  ],
  "face_detections": [
    {
      "detection_id": "face-1",
      "bbox": { "x": 500, "y": 300, "width": 200, "height": 250 },
      "confidence": 0.98
    }
  ]
}
```

### Step 3: Anonymize Selected Regions

```bash
curl -X POST "http://localhost:8000/anonymize" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "abc-123",
    "regions": [
      {
        "bbox": {"x": 100, "y": 200, "width": 150, "height": 30},
        "pii_type": "phone"
      },
      {
        "bbox": {"x": 500, "y": 300, "width": 200, "height": 250},
        "pii_type": "face"
      }
    ],
    "method": "generate"
  }'
```

Response:

```json
{
  "original_image_id": "abc-123",
  "anonymized_image_id": "xyz-789",
  "success": true,
  "message": "Successfully anonymized 2 regions",
  "processed_count": 2
}
```

### Step 4: Download Anonymized Image

```bash
# Use the anonymized_image_id from the response
curl -X GET "http://localhost:8000/download/xyz-789" \
  -o anonymized.png
```

### Step 5: Clean Up (Optional)

```bash
curl -X DELETE "http://localhost:8000/clear/abc-123"
```

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found (image_id not found, no detections, etc.)
- `500`: Internal server error

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
GEMINI_API_KEY=your_google_gemini_api_key_here

# Optional (defaults shown)
DEFAULT_FACE_METHOD=blur
DEFAULT_TEXT_METHOD=generate
```

### Detection Settings

Edit `src/config.py`:

```python
# Model Settings
DETECTION_MODEL = "gemini-3-flash-preview"
IMAGEN_MODEL = "gemini-2.5-flash-image"

# Detection Confidence Thresholds
MIN_FACE_CONFIDENCE = 0.7
MIN_TEXT_CONFIDENCE = 0.8

# Image Generation Settings
MASK_PADDING = 10  # Padding around masked regions in pixels
```

---

## Anonymization Methods

### 1. GENERATE (AI-Powered)

Uses Google Gemini's generative AI to intelligently fill detected regions with contextually appropriate content.

**Best for:**

- Text PII (replaces with realistic placeholder text)
- Faces (generates different face maintaining scene context)
- Complex backgrounds

**Example:**

```json
{ "method": "generate" }
```

### 2. BLUR

Applies Gaussian blur to the detected region.

**Best for:**

- Quick anonymization
- Faces when context preservation isn't critical
- Low-latency requirements

**Example:**

```json
{ "method": "blur" }
```

### 3. BLACK_BOX

Overlays a solid black rectangle on the region.

**Best for:**

- Maximum privacy (complete obscuration)
- Low computational cost
- Simple redaction

**Example:**

```json
{ "method": "black_box" }
```

---

## Rate Limits & Performance

- **Detection**: ~2-5 seconds per image (depends on complexity)
- **Anonymization (GENERATE)**: ~3-8 seconds per region (batched processing)
- **Anonymization (BLUR/BLACK_BOX)**: <100ms per region

**Note**: The API uses Google Gemini API which has its own rate limits. For production use, implement proper rate limiting and caching strategies.

---

## Security Considerations

1. **API Key Protection**: Never expose your `GEMINI_API_KEY` in client-side code
2. **Input Validation**: All uploaded files are validated for image format
3. **CORS**: Configure `allow_origins` in production (currently set to `["*"]` for development)
4. **Cache Management**: Regularly clear cache to prevent disk space issues
5. **File Size Limits**: Configure appropriate upload size limits for production

---

## Development

### Running the Server

```bash
# Using uvicorn directly
uvicorn src.api:app --reload

# Using the main script
uv run ./main.py
```

### Running Tests

```bash
# Lint code
ruff check

# Format code
ruff format
```

---

## License

[Your License Here]

## Support

For issues and questions, please contact [Your Contact Info]
