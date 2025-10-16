# minimal comments
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

def to_numpy(pil_image):
    return np.array(pil_image.convert("RGB"))

def draw_boxes_pil(image_np, detections, class_names_map):
    img = Image.fromarray(image_np.astype('uint8'))
    draw = ImageDraw.Draw(img)
    try:
        # Load a default font for better cross-platform compatibility
        # DejaVuSans.ttf might not exist everywhere.
        font = ImageFont.truetype("arial.ttf", 14) 
    except Exception:
        font = ImageFont.load_default()
    
    # Define a color map for visual distinction (optional)
    COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange"] 

    for i, d in enumerate(detections):
        bbox = d["bbox"]
        cid = d["class_id"]
        conf = d["conf"]
        
        # Capitalize the label as requested
        label = class_names_map.get(cid, str(cid)).upper()
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Use a distinct color based on the class index
        color = COLORS[cid % len(COLORS)] 

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        text = f"{label} {conf:.2f}"
        
        # --- FIX: Replaced deprecated textsize() with textlength() ---
        
        # Get width of the text using textlength()
        text_width = draw.textlength(text, font=font)
        # Assuming a fixed height for simplicity, or use font.getbbox() to be more accurate
        text_height = font.getbbox("T")[3] - font.getbbox("T")[1] + 4 

        # background rectangle (adjusted for the new height calculation)
        draw.rectangle([x1, y1 - text_height, x1 + text_width + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height + 2), text, fill="white", font=font)
        
    return np.array(img)

def pil_bytes_from_numpy(image_np, fmt="PNG"):
    img = Image.fromarray(image_np.astype('uint8'))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()