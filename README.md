# SmartTrolley-Backend

A Python backend for the **Smart Trolley System**, providing real-time person detection, multi-object tracking, person re-identification, and customer registration using **YOLO11**, **ByteTrack**, and **TorchReID (OSNet)**.

---

## Features

- Real-time video streaming
- YOLO11 object detection
- ByteTrack multi-object tracking
- Person Re-Identification (TorchReID - OSNet)
- Customer registration from captured images
- WebSocket communication
- REST API backend

---

## Project Structure

```
SmartTrolley-Backend/
│── server.py
│── my_bytetrack.py
│── requirements.txt
│── .gitignore
│── yolo11n.pt
└── README.md
```

---

## Requirements

- Python 3.10 or later
- Git
- CUDA-enabled GPU (recommended)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/krisjk003/SmartTrolley-Backend.git
cd SmartTrolley-Backend
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Clone and install TorchReID

TorchReID is required for person re-identification.

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -e .
cd ..
```

---

## Running the Server

```bash
python server.py
```

---

## Dependencies

- PyTorch
- TorchVision
- Ultralytics YOLO11
- OpenCV
- ByteTrack
- TorchReID (OSNet)
- WebSockets

---

## Notes

- Ensure `yolo11n.pt` is available in the project directory before running the server.
- Install TorchReID before starting the backend.
- A CUDA-enabled GPU is recommended for optimal real-time performance.

---



## Author

**Jyothish Kumar JS**

GitHub: https://github.com/krisjk003
