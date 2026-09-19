import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
from inference import screen_cow_video, load_gaitguard_model

PORT = 8000
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Preload model globally for instant inference
g_model = None
g_device = None
g_config = None

def get_model():
    global g_model, g_device, g_config
    if g_model is None and os.path.exists("models/best_model.pth"):
        try:
            g_model, g_device, g_config = load_gaitguard_model()
            print("GaitGuard model loaded successfully into memory.")
        except Exception as e:
            print("Model load warning:", e)
    return g_model, g_device, g_config

class GaitGuardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            with open("static/index.html", "rb") as f:
                self.wfile.write(f.read())
            return
            
        elif parsed.path.startswith("/static/"):
            filepath = parsed.path.lstrip("/")
            if os.path.exists(filepath):
                self.send_response(200)
                if filepath.endswith(".css"):
                    self.send_header("Content-type", "text/css")
                elif filepath.endswith(".js"):
                    self.send_header("Content-type", "application/javascript")
                elif filepath.endswith(".png"):
                    self.send_header("Content-type", "image/png")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return
                
        elif parsed.path == "/api/samples":
            # Provide sample test videos for 1-click judging demo
            samples = []
            test_videos = [
                ("Normal Locomotion - Sample 1", r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Normal\N (1).mp4", "NORMAL"),
                ("Normal Locomotion - Sample 2", r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Normal\N (3).mp4", "NORMAL"),
                ("Lame Locomotion - Sample 1", r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Lame\L (1).mp4", "LAME"),
                ("Lame Locomotion - Sample 2", r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Lame\L (5).mp4", "LAME")
            ]
            for name, p, expected in test_videos:
                if os.path.exists(p):
                    samples.append({
                        "name": name,
                        "path": p,
                        "expected": expected,
                        "filename": os.path.basename(p)
                    })
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(samples).encode("utf-8"))
            return

        elif parsed.path.startswith("/video/"):
            # Serve video for playback preview
            video_name = urllib.parse.unquote(parsed.path.replace("/video/", ""))
            target_path = os.path.join(UPLOAD_DIR, video_name)
            if not os.path.exists(target_path):
                l_path = os.path.join(r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Lame", video_name)
                n_path = os.path.join(r"C:\Users\anshu\Downloads\CattleLameness-main\CattleLameness-main\Data\Normal", video_name)
                if os.path.exists(l_path):
                    target_path = l_path
                elif os.path.exists(n_path):
                    target_path = n_path
                    
            if os.path.exists(target_path):
                self.send_response(200)
                self.send_header("Content-type", "video/mp4")
                self.send_header("Content-Length", str(os.path.getsize(target_path)))
                self.end_headers()
                with open(target_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/screen":
            content_type = self.headers.get("Content-Type", "")
            video_path = None
            
            if "multipart/form-data" in content_type:
                content_len = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_len)
                boundary_token = content_type.split("boundary=")[-1].strip().encode()
                parts = raw_body.split(b"--" + boundary_token)
                
                for part in parts:
                    if b'filename="' in part:
                        headers_part, file_data = part.split(b"\r\n\r\n", 1)
                        file_data = file_data.rstrip(b"\r\n--")
                        save_path = os.path.join(UPLOAD_DIR, "uploaded_cow.mp4")
                        with open(save_path, "wb") as f:
                            f.write(file_data)
                        video_path = save_path
                        break
                    elif b'name="sample_path"' in part:
                        headers_part, field_data = part.split(b"\r\n\r\n", 1)
                        video_path = field_data.rstrip(b"\r\n--").decode("utf-8").strip()
                        break
                        
            elif "application/json" in content_type:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8")
                data = json.loads(body)
                video_path = data.get("video_path")
                
            if not video_path or not os.path.exists(video_path):
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid video path: {video_path}"}).encode("utf-8"))
                return
                
            model, device, config = get_model()
            try:
                result = screen_cow_video(video_path, model=model, device=device)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

def start_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, GaitGuardHandler)
    print(f"GaitGuard AI Web Server active on http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    start_server()
