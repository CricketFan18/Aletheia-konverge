import cv2
import numpy as np
import base64
from PIL import Image, ImageChops, ImageEnhance
import io

def generate_ela_heatmap(image_path: str, quality: int = 90) -> str:
    """
    Generates an Error Level Analysis heatmap and returns it as a Base64 string
    so the React frontend can immediately render it as an <img src="...">.
    """
    try:
        original = Image.open(image_path).convert('RGB')
        
        # Save a temporary compressed version to memory
        temp_io = io.BytesIO()
        original.save(temp_io, 'JPEG', quality=quality)
        temp_io.seek(0)
        compressed = Image.open(temp_io)
        
        # Calculate the absolute difference between original and compressed
        ela_image = ImageChops.difference(original, compressed)
        
        # Enhance the difference to make it visually pop as a "heatmap"
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Convert to a color map (Heatmap style: Jet or Inferno)
        ela_cv = cv2.cvtColor(np.array(ela_image), cv2.COLOR_RGB2BGR)
        ela_cv = cv2.applyColorMap(ela_cv, cv2.COLORMAP_INFERNO)
        
        # Encode back to Base64 for the API response
        _, buffer = cv2.imencode('.jpg', ela_cv)
        base64_heatmap = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{base64_heatmap}"
    except Exception as e:
        print(f"[ELA ERROR] {e}")
        return None