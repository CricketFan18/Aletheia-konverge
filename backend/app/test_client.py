import requests
import json
import os
import base64

API_URL = "http://127.0.0.1:8000/api/analyze"

def test_authenticity_api(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Could not find '{image_path}'")
        return

    print(f"🚀 Sending '{image_path}' to the ML Engine...")

    with open(image_path, "rb") as image_file:
        files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
        
        try:
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                print("\n✅ API MATCH SUCCESS! Payload received.\n")
                payload = response.json()
                
                # 1. Save the heatmap to your disk so you can look at it
                if "heatmap_image" in payload["evidence"] and payload["evidence"]["heatmap_image"]:
                    base64_str = payload["evidence"]["heatmap_image"]
                    # Strip off the "data:image/jpeg;base64," prefix to get the raw data
                    raw_b64 = base64_str.split(",")[1] 
                    
                    with open("heatmap_result.jpg", "wb") as fh:
                        fh.write(base64.b64decode(raw_b64))
                    print("🔥 Heatmap saved to your folder as 'heatmap_result.jpg'! Go open it!")

                # 2. Truncate for the terminal printout
                if "heatmap_image" in payload["evidence"]:
                    payload["evidence"]["heatmap_image"] = base64_str[:40] + "...[TRUNCATED]..."
                
                print(json.dumps(payload, indent=4))
            else:
                print(f"\n❌ API REJECTED. Status Code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("\n❌ CONNECTION FAILED: Is Uvicorn running?")

if __name__ == "__main__":
    TARGET_IMAGE = "../saved_images/modiREAL.jpg" # Adjust if needed
    test_authenticity_api(TARGET_IMAGE)