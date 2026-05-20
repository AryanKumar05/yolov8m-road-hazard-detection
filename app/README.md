# Inference Demo App

A lightweight web application for testing model weights in the browser.

## Supported Weight Formats
- `.pt` — PyTorch checkpoint (direct Ultralytics)
- `.onnx` — ONNX Runtime (cross-platform)
- `.engine` — TensorRT (NVIDIA only, fastest)

## Features
- Drag-and-drop weight upload at runtime
- Webcam live inference with real-time bounding boxes
- Video file inference with pothole/hump proximity alerts
- Distance estimation via camera focal length + mounting height
- Confidence threshold slider

## Running

```bash
pip install fastapi uvicorn python-multipart
python app/backend.py --port 8080
# Open http://localhost:8080
```
