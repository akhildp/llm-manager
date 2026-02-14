"""Chart pattern detector using foduucom/stockmarket-pattern-detection-yolov8.

Uses YOLOv8 object detection to identify specific trading chart patterns:
- Head and shoulders bottom / top
- M_Head (double top)
- W_Bottom (double bottom)
- Triangle
- StockLine
"""

import base64
import io
import logging
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger("chart_recognizer")

MODEL_REPO = "foduucom/stockmarket-pattern-detection-yolov8"


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the YOLOv8 chart pattern detection model."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    logger.info("Downloading YOLOv8 chart pattern model from HuggingFace...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.pt")
    logger.info(f"Loading YOLOv8 model from {model_path}")
    model = YOLO(model_path)
    logger.info(f"YOLOv8 chart pattern model loaded. Classes: {model.names}")
    return model


def classify_chart(image_base64: str, top_k: int = 5) -> dict:
    """
    Detect chart patterns in an image using YOLOv8.

    Args:
        image_base64: Base64-encoded image (with or without data URI prefix).
        top_k: Maximum number of detections to return.

    Returns:
        dict with 'patterns' (list of {label, probability, bbox}) and 'summary' (text).
    """
    model = _load_model()

    # Strip data URI prefix if present
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    # Decode and open image
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    # Run detection
    results = model(img_array, verbose=False)

    patterns = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(min(len(boxes), top_k)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = result.names.get(cls_id, f"class_{cls_id}")
                bbox = boxes.xyxy[i].tolist()

                # Normalize bbox
                width, height = image.size
                bbox_norm = [
                    round(bbox[0] / width, 3),
                    round(bbox[1] / height, 3),
                    round(bbox[2] / width, 3),
                    round(bbox[3] / height, 3),
                ]

                patterns.append({
                    "label": label,
                    "probability": round(conf * 100, 1),
                    "bbox": [round(v, 1) for v in bbox],
                    "bbox_norm": bbox_norm,
                })

    # Sort by confidence
    patterns.sort(key=lambda x: x["probability"], reverse=True)

    # Generate annotated image if patterns found (using PIL for reliability)
    annotated_b64 = None
    if patterns:
        try:
            from PIL import ImageDraw, ImageFont
            annotated_img = image.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # Load font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except OSError:
                logger.warning("DejaVuSans-Bold not found, using default font.")
                font = ImageFont.load_default()

            for p in patterns:
                # Cast bbox to int for PIL drawing
                bbox = [int(v) for v in p["bbox"]]
                label = p["label"]
                prob = p["probability"]
                color = "#00FF00" # Lime Green
                
                # Draw thicker box
                draw.rectangle(bbox, outline=color, width=4)
                
                # Draw label
                text = f"{label} {prob}%"
                
                # Measure text
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                    text_width = right - left
                    text_height = bottom - top
                except AttributeError:
                    # Fallback for old PIL
                    text_width = len(text) * 12
                    text_height = 20
                
                # Draw text background above the box
                text_bg = [bbox[0], bbox[1] - text_height - 10, bbox[0] + text_width + 10, bbox[1]]
                # If box is at top edge, draw background INSIDE
                if bbox[1] < text_height + 10:
                    text_bg = [bbox[0], bbox[1], bbox[0] + text_width + 10, bbox[1] + text_height + 10]
                    text_pos = (bbox[0] + 5, bbox[1] + 5)
                else:
                    text_pos = (bbox[0] + 5, bbox[1] - text_height - 5)
                
                draw.rectangle(text_bg, fill="black")
                draw.text(text_pos, text, fill=color, font=font)

            buffered = io.BytesIO()
            annotated_img.save(buffered, format="JPEG", quality=85)
            annotated_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to draw annotations: {e}")
            patterns.append({"label": f"IMG ERR: {str(e)[:20]}", "probability": 0, "bbox": [0,0,0,0], "bbox_norm": [0,0,0,0]})

    # Build summary
    if patterns:
        summary_lines = ["📊 **Chart Patterns Detected**"]
        for p in patterns:
            summary_lines.append(f"  • {p['label']}: {p['probability']}% confidence")
        summary = "\n".join(summary_lines)
    else:
        summary = "ℹ️ No specific chart patterns detected in this image."

    return {
        "patterns": patterns,
        "summary": summary,
        "is_chart": len(patterns) > 0,
        "annotated_image": annotated_b64,
    }
