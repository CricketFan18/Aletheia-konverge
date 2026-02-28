import requests
import json
import os
import base64

API_URL = "http://127.0.0.1:8000/api/analyze"

def test_authenticity_api(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Could not find '{image_path}'. Please check your path.")
        return

    print(f"🚀 Sending '{image_path}' to the ML Engine...")

    with open(image_path, "rb") as image_file:
        files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
        
        try:
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                print("\n✅ API MATCH SUCCESS! Payload received.\n")
                payload = response.json()
                
                # Safely extract evidence to avoid KeyErrors
                evidence = payload.get("evidence", {})
                heatmap_b64 = evidence.get("heatmap_image")
                
                # Save the heatmap
                if heatmap_b64:
                    # Strip prefix if it exists
                    raw_b64 = heatmap_b64.split(",")[1] if "," in heatmap_b64 else heatmap_b64
                    
                    with open("heatmap_result.jpg", "wb") as fh:
                        fh.write(base64.b64decode(raw_b64))
                    print("🔥 Heatmap saved to your folder as 'heatmap_result.jpg'! Go open it!")

                    # 2. Truncate for the terminal printout
                    payload["evidence"]["heatmap_image"] = heatmap_b64[:40] + "...[TRUNCATED]..."
                else:
                    print("⚠️ Note: No heatmap was returned by the server.")
                
                print(json.dumps(payload, indent=4))
            else:
                print(f"\n❌ API REJECTED. Status Code: {response.status_code}")
                print(f"Error Details: {response.text}") # Prints the actual error from FastAPI
                
        except requests.exceptions.ConnectionError:
            print("\n❌ CONNECTION FAILED: Is Uvicorn running? Run 'uvicorn main:app --reload' first.")

if __name__ == "__main__":
    # Ensure this relative path is correct relative to where you run the script!
    TARGET_IMAGE = "../saved_images/modiAI.png" 
    test_authenticity_api(TARGET_IMAGE)