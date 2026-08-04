"""
===============================================================================
ULTRA-LOW LATENCY H.264 WEBSOCKET RECEIVER, YOLO11 TRACKING & ESP32 ROBOT SERVER
WITH AUTOMATIC ANDROID REFERENCE TRANSFER & OPTIMIZED OSNET HYBRID ReID
===============================================================================
Optimized for Low-Latency Android MediaCodec -> PyAV -> YOLO11 ByteTrack Streams
with Integrated ESP32 HTTP Robot Following, Dead-Zone Steering, and OSNet ReID.
===============================================================================
"""

import asyncio
import base64
import json
import logging
import queue
import threading
import time
import urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Set, Tuple
import warnings

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import websockets
from ultralytics import YOLO



warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*'half' is deprecated.*"
)

# Import TorchReID Feature Extractor for OSNet
try:
    from torchreid.utils import FeatureExtractor
    print("FeatureExtractor imported successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Actual import error:", e)
    FeatureExtractor = None

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("H264RobotServer")

# =============================================================================
# TRACKING & STREAM CONFIGURATION
# =============================================================================
HOST = "0.0.0.0"
PORT = 8765
YOLO_MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "bytetrack.yaml"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
MAX_FRAME_AGE_SEC = 0.100  # 100 ms budget to accommodate Wi-Fi jitter without over-dropping
PERSON_CLASS_ID = 0        # COCO class index for 'person'

# =============================================================================
# CENTRALIZED ReID & REGISTRATION CONFIGURATION
# =============================================================================
REID_CONFIG = {
    "model_name": "osnet_x0_5",           # Lightweight model: 0.27 GFLOPs vs 0.98 for osnet_x1_0
    "sim_threshold": 0.5,                # Minimum cosine similarity to candidate
    "search_interval_sec": 0.15,          # Throttle ReID searches to every 350 ms
    "cache_ttl_sec": 2.0,                 # Time-to-live for cached track embeddings
    "max_cache_size": 50,                 # LRU cache upper boundary to guarantee stable memory
    "hysteresis_frames": 1,               # Consecutive positive matches required to lock target
    "target_lost_timeout_sec": 5.0,       # Hold lock for 2 seconds when occluded before release
    "registration_port": 8080,            # REST endpoint port for Android customer registration
    "distance_metric": "cosine"           # Future-proofing: 'cosine' or 'euclidean'
}

DEBUG_CONFIG = {
    "enable_manual_mouse_override": True, # Allow mouse-click locking for testing
    "enable_keyboard_shortcuts": True,    # Allow 'c', 'r', '1'-'9' debug overrides
    "show_hud_stats": True                # Display structured debug metrics on frame HUD
}

# =============================================================================
# ESP32 ROBOT CONTROL CONFIGURATION
# =============================================================================
ESP32_IP = "192.168.4.1"          # Target ESP32 IP Address
ESP32_PORT = 80                   # HTTP Commander Port
DEAD_ZONE_RATIO = 0.15            # +/- 15% from frame center is considered "Straight (/F)"
COMMAND_MIN_INTERVAL = 0.05       # Minimum 50ms between requests (max 20 Hz rate limit)
COMMAND_KEEPALIVE_INTERVAL = 0.5  # Resend command every 500ms if unchanged (watchdog keep-alive)


# =============================================================================
# MODULAR LAYER: PERSON RE-IDENTIFICATION (OSNET) MANAGER
# =============================================================================
class PersonReIDManager:
    """
    Production Person Re-Identification using OSNet (TorchReID).
    
    Features:
      - Multi-customer registry support (customers[customer_id] -> embedding).
      - Strict registration validation: verifies person detection and embedding generation.
      - LRU track-ID cache with Time-to-Live (TTL) and Maximum Capacity limits.
      - Throttled search intervals and multi-frame hysteresis to prevent flicker.
      - Bypassed automatically when ByteTrack holds an active target lock.
    """
    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.sim_threshold = config.get("sim_threshold", 0.65)
        self.search_interval = config.get("search_interval_sec", 0.35)
        self.cache_ttl = config.get("cache_ttl_sec", 2.0)
        self.max_cache_size = config.get("max_cache_size", 50)
        self.hysteresis_target = config.get("hysteresis_frames", 3)

        # Multi-customer embedding registry: customer_id -> unified reference embedding
        self.customers: Dict[str, torch.Tensor] = {}
        self.active_customer_id: str = "default"
        self.is_registered: bool = False

        # LRU Track Cache: OrderedDict[track_id -> (embedding_tensor, last_computed_timestamp)]
        self.embedding_cache: OrderedDict[int, Tuple[torch.Tensor, float]] = OrderedDict()
        self.cache_lock = threading.Lock()

        # Search Throttling & Hysteresis State
        self.last_search_time: float = 0.0
        self.hysteresis_candidate_id: Optional[int] = None
        self.hysteresis_count: int = 0

        if FeatureExtractor is None:
            logger.error("torchreid is not installed! Run: pip install torchreid")
            self.extractor = None
        else:
            model_name = config.get("model_name", "osnet_x0_5")
            logger.info(f"Initializing OSNet Feature Extractor ({model_name}) on {device.upper()}...")
            self.extractor = FeatureExtractor(
                model_name=model_name,
                device=device
            )
            self.warmup()

    def warmup(self):
        """Runs dummy inputs through OSNet to compile CUDA kernels before live streaming."""
        if self.extractor is None:
            return
        logger.info("Running OSNet CUDA warm-up iterations...")
        dummy_crops = [np.zeros((256, 128, 3), dtype=np.uint8) for _ in range(3)]
        try:
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                _ = self.extractor(dummy_crops)
            logger.info("OSNet warm-up complete.")
        except Exception as e:
            logger.warning(f"OSNet warm-up encountered non-fatal issue: {e}")

    def register_customer_from_images(
        self,
        yolo_model: YOLO,
        bgr_images: List[np.ndarray],
        customer_id: str = "default"
    ) -> Tuple[bool, str]:
        """
        Validates reference images, detects person ROIs via YOLO11, generates embeddings,
        averages them into a single vector, and registers the customer.
        """
        if self.extractor is None:
            return False, "OSNet feature extractor is not initialized."

        if len(bgr_images) < 2:
            msg = f"Registration requires at least 2 reference images; received {len(bgr_images)}."
            logger.error(msg)
            return False, msg

        raw_embeddings = []
        logger.info(f"Processing {len(bgr_images)} reference images for customer '{customer_id}'...")

        for idx, img in enumerate(bgr_images, 1):
            if img is None or img.size == 0:
                msg = f"Reference image #{idx} is empty or corrupted."
                logger.error(msg)
                return False, msg

            # Detect person ROI in reference photo
            results = yolo_model.predict(source=img, conf=0.45, verbose=False)
            best_crop = None
            max_area = 0

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box, cls_id in zip(boxes.xyxy.cpu().numpy(), boxes.cls.int().cpu().numpy()):
                    if int(cls_id) == PERSON_CLASS_ID:
                        x1, y1, x2, y2 = map(int, box)
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        if area > max_area:
                            max_area = area
                            best_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]

            if best_crop is None or best_crop.size == 0:
                msg = f"Registration Failed: No person detected in reference image #{idx}."
                logger.error(msg)
                return False, msg

            emb = self._compute_embedding(best_crop)
            if emb is None:
                msg = f"Registration Failed: Could not generate OSNet embedding for image #{idx}."
                logger.error(msg)
                return False, msg

            raw_embeddings.append(emb)
            logger.info(f"Successfully validated and embedded reference image #{idx}.")

        # Average all reference embeddings into a single L2-normalized vector
        stacked = torch.cat(raw_embeddings, dim=0)
        averaged = torch.mean(stacked, dim=0, keepdim=True)
        unified_embedding = F.normalize(averaged, p=2, dim=1)

        self.customers[customer_id] = unified_embedding
        self.active_customer_id = customer_id
        self.is_registered = True

        with self.cache_lock:
            self.embedding_cache.clear()
        self.reset_hysteresis()

        success_msg = f"Customer '{customer_id}' registered successfully with {len(raw_embeddings)} reference images."
        logger.info(success_msg)
        return True, success_msg

    def _compute_embedding(self, crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """Extracts a 512-dim L2-normalized OSNet feature vector from a BGR image crop."""
        if self.extractor is None or crop_bgr.size == 0:
            return None
        try:
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                features = self.extractor([crop_bgr])
            if isinstance(features, torch.Tensor):
                embedding = features.to(self.device)
            else:
                embedding = torch.tensor(features, device=self.device)
            return F.normalize(embedding, p=2, dim=1)
        except Exception as e:
            logger.error(f"Error computing OSNet embedding: {e}")
            return None

    def reset_hysteresis(self):
        """Resets multi-frame lock confirmation counters."""
        self.hysteresis_candidate_id = None
        self.hysteresis_count = 0

    def prune_cache(self, active_track_ids: Set[int], now: float):
        """Evicts stale track IDs via TTL and enforces LRU max capacity boundaries."""
        with self.cache_lock:
            # TTL & track existence pruning
            stale_keys = [
                tid for tid, (_, ts) in self.embedding_cache.items()
                if (tid not in active_track_ids) or (now - ts > self.cache_ttl)
            ]
            for key in stale_keys:
                del self.embedding_cache[key]

            # LRU capacity pruning
            while len(self.embedding_cache) > self.max_cache_size:
                self.embedding_cache.popitem(last=False)

    def identify_target(
        self,
        frame: np.ndarray,
        candidate_boxes: List[Tuple[int, Tuple[int, int, int, int]]],
        active_track_ids: Set[int]
    ) -> Optional[Tuple[int, float]]:
        """
        Throttled, LRU-cached, and hysteresis-smoothed ReID search against the
        active customer's unified reference embedding.
        """
        if not self.is_registered or len(candidate_boxes) == 0:
            return None

        target_embedding = self.customers.get(self.active_customer_id)
        if target_embedding is None:
            return None

        now = time.time()
        if now - self.last_search_time < self.search_interval:
            return None
        self.last_search_time = now

        self.prune_cache(active_track_ids, now)

        best_match_id = None
        highest_sim = -1.0
        frame_h, frame_w = frame.shape[:2]

        for track_id, (x1, y1, x2, y2) in candidate_boxes:
            cand_embedding = None

            with self.cache_lock:
                if track_id in self.embedding_cache:
                    cached_emb, ts = self.embedding_cache[track_id]
                    if now - ts <= self.cache_ttl:
                        cand_embedding = cached_emb
                        self.embedding_cache.move_to_end(track_id)

            if cand_embedding is None:
                crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                if crop.size == 0:
                    continue
                cand_embedding = self._compute_embedding(crop)
                if cand_embedding is not None:
                    with self.cache_lock:
                        self.embedding_cache[track_id] = (cand_embedding, now)
                        self.embedding_cache.move_to_end(track_id)

            if cand_embedding is None:
                continue

            sim = F.cosine_similarity(cand_embedding, target_embedding).item()
            if sim > highest_sim:
                highest_sim = sim
                best_match_id = track_id

        if best_match_id is not None and highest_sim >= self.sim_threshold:
            logger.info(
              f"Best Match: ID={best_match_id}, Similarity={highest_sim:.3f}, Threshold={self.sim_threshold}"
            )
            if best_match_id == self.hysteresis_candidate_id:
                self.hysteresis_count += 1
            else:
                self.hysteresis_candidate_id = best_match_id
                self.hysteresis_count = 1

            if self.hysteresis_count >= self.hysteresis_target:
                logger.info(f"OSNet ReID Match Confirmed -> Lock ID:{best_match_id} (Sim: {highest_sim:.2f})")
                self.reset_hysteresis()
                return best_match_id, highest_sim
        else:
            self.reset_hysteresis()

        return None


# =============================================================================
# MODULAR LAYER: ANDROID HTTP REGISTRATION SERVER
# =============================================================================
class CustomerRegistrationHandler(BaseHTTPRequestHandler):
    """HTTP REST Handler accepting Base64-encoded customer reference JPEGs from Android."""
    reid_manager: Optional[PersonReIDManager] = None
    yolo_model: Optional[YOLO] = None

    def do_POST(self):
        if self.path == "/register":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode('utf-8'))
                images_b64 = payload.get("images", [])
                customer_id = payload.get("customer_id", "default")

                bgr_images = []
                for idx, b64_str in enumerate(images_b64):
                    img_bytes = base64.b64decode(b64_str)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if bgr_img is not None:
                        bgr_images.append(bgr_img)

                success, message = self.reid_manager.register_customer_from_images(
                    yolo_model=self.yolo_model,
                    bgr_images=bgr_images,
                    customer_id=customer_id
                )

                self.send_response(200 if success else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response_payload = {"success": success, "message": message}
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
            except Exception as e:
                logger.error(f"Error handling Android registration request: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "message": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default HTTP console spam


def start_registration_server(reid_manager: PersonReIDManager, yolo_model: YOLO, port: int):
    """Launches the lightweight HTTP registration server in a daemon thread."""
    CustomerRegistrationHandler.reid_manager = reid_manager
    CustomerRegistrationHandler.yolo_model = yolo_model
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, CustomerRegistrationHandler)
    logger.info(f"Android REST Registration Server listening on http://0.0.0.0:{port}/register")
    httpd.serve_forever()


# =============================================================================
# EXISTING BENCHMARK INFRASTRUCTURE (UNMODIFIED)
# =============================================================================
class InstantFrameBuffer:
    """Zero-latency thread-safe frame buffer using Threading Event signaling."""
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0

    def put(self, frame: np.ndarray, timestamp: float = 0.0) -> None:
        with self._lock:
            self._frame = frame
            self._timestamp = timestamp
            self._event.set()

    def get(self, timeout: float = 0.01) -> Tuple[Optional[np.ndarray], float]:
        if self._event.wait(timeout=timeout):
            with self._lock:
                frame = self._frame
                ts = self._timestamp
                self._frame = None
                self._event.clear()
                return frame, ts
        return None, 0.0


class ESP32HTTPCommander:
    """Background HTTP command sender with queue overwrite, rate limiting, and keep-alive."""
    def __init__(self, ip: str, port: int = 80):
        self.base_url = f"http://{ip}:{port}"
        self.cmd_queue: queue.Queue = queue.Queue(maxsize=1)
        self.last_sent_cmd: Optional[str] = None
        self.last_send_time: float = 0.0
        self._running = True
        
        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="ESP32CommanderThread"
        )
        self.thread.start()
        logger.info(f"ESP32 HTTP Commander initialized -> {self.base_url}")

    def send_command(self, cmd: str) -> None:
        try:
            self.cmd_queue.put_nowait(cmd)
        except queue.Full:
            try:
                self.cmd_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.cmd_queue.put_nowait(cmd)
            except queue.Full:
                pass

    def _send_http(self, cmd: str) -> None:
        url = f"{self.base_url}/{cmd.lstrip('/')}"
        try:
            with urllib.request.urlopen(url, timeout=0.2) as response:
                _ = response.read()
            self.last_sent_cmd = cmd
            self.last_send_time = time.time()
        except Exception as e:
            logger.debug(f"ESP32 HTTP Command ('{cmd}') failed: {e}")

    def _worker_loop(self) -> None:
        while self._running:
            try:
                cmd = self.cmd_queue.get(timeout=0.02)
            except queue.Empty:
                now = time.time()
                if (self.last_sent_cmd is not None and 
                    now - self.last_send_time >= COMMAND_KEEPALIVE_INTERVAL):
                    self._send_http(self.last_sent_cmd)
                continue

            now = time.time()
            if (cmd == self.last_sent_cmd and 
                now - self.last_send_time < COMMAND_MIN_INTERVAL):
                continue
            
            self._send_http(cmd)

    def stop(self) -> None:
        self._running = False
        self.send_command("S")
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)


# Global state
frame_buffer = InstantFrameBuffer()
mouse_click_coords: Optional[Tuple[int, int]] = None
locked_target_id: Optional[int] = None
locked_target_last_seen: float = 0.0
last_similarity_score: float = 0.0
tracking_mode_label: str = "OFFLINE"


def on_mouse_click(event, x, y, flags, param):
    """OpenCV Mouse Callback for optional debug target locking override."""
    global mouse_click_coords
    if DEBUG_CONFIG["enable_manual_mouse_override"] and event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_coords = (x, y)


class LowLatencyH264Decoder:
    """PyAV H.264 decoder context configured for zero frame delay."""
    def __init__(self):
        self.codec = None
        self._init_codec()

    def _init_codec(self):
        if self.codec is not None:
            try:
                self.codec.close()
            except Exception:
                pass

        self.codec = av.CodecContext.create('h264', 'r')
        self.codec.thread_type = 'SLICE'
        self.codec.thread_count = 1
        self.codec.options = {
            'flags': 'low_delay',
            'fflags': 'nobuffer',
            'flags2': 'fast'
        }

    def decode_packet(self, packet_bytes: bytes) -> List[np.ndarray]:
        decoded_frames = []
        try:
            packets = self.codec.parse(packet_bytes)
            for packet in packets:
                frames = self.codec.decode(packet)
                for frame in frames:
                    decoded_frames.append(frame.to_ndarray(format='rgb24'))
        except (av.InvalidDataError, av.FFmpegError) as e:
            logger.warning(f"Non-fatal packet error ignored: {e}")
        except av.CodecError as e:
            logger.error(f"Fatal PyAV CodecError: {e}. Reinitializing decoder...")
            self._init_codec()
        except Exception as e:
            logger.warning(f"Unhandled non-fatal decoder exception ignored: {e}")
            
        return decoded_frames

    def reset(self):
        self._init_codec()


async def websocket_handler(websocket, path=None):
    """WebSocket server handler receiving raw H.264 binary packets."""
    client_address = websocket.remote_address
    logger.info(f"Android client connected: {client_address}")
    decoder = LowLatencyH264Decoder()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                recv_time = time.time()
                frames = decoder.decode_packet(message)
                for frame in frames:
                    frame_buffer.put(frame, recv_time)

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Android client disconnected: {client_address}")
    finally:
        decoder.reset()


def start_websocket_server(loop):
    """Runs WebSocket server loop in background thread."""
    asyncio.set_event_loop(loop)
    server = websockets.serve(
        websocket_handler,
        HOST,
        PORT,
        max_queue=1,
        max_size=None,
        ping_interval=20,
        ping_timeout=10
    )
    logger.info(f"H.264 WebSocket server listening on ws://{HOST}:{PORT}")
    loop.run_until_complete(server)
    loop.run_forever()


# =============================================================================
# MAIN EXECUTABLE & HYBRID TRACKING LOOP
# =============================================================================
def main():
    global locked_target_id, mouse_click_coords, locked_target_last_seen
    global last_similarity_score, tracking_mode_label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #use_half = (device == "cuda")
    use_half = False

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    logger.info(f"Loading YOLO11 model on {device.upper()} (FP16: {use_half})...")
    model = YOLO(YOLO_MODEL_PATH)

    if device == "cuda":
        model.to("cuda")
        model.fuse()

    # Warm up YOLO11 detection AND ByteTrack initialization at 720p
    logger.info("Running CUDA & ByteTrack warm-up iterations...")
    dummy_input = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(3):
        # Using modern recommended inference syntax without deprecated arguments
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_half):
            model.track(
                source=dummy_input,
                persist=True,
                tracker=TRACKER_CONFIG,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=device,
                half=False,
                verbose=False
            )

    # -------------------------------------------------------------------------
    # INITIALIZE OSNET ReID MANAGER & ANDROID REST REGISTRATION SERVER
    # -------------------------------------------------------------------------
    reid_manager = PersonReIDManager(config=REID_CONFIG, device=device)

    reg_thread = threading.Thread(
        target=start_registration_server,
        args=(reid_manager, model, REID_CONFIG["registration_port"]),
        daemon=True,
        name="AndroidRegistrationThread"
    )
    reg_thread.start()

    # Initialize Background ESP32 Robot Commander
    robot_commander = ESP32HTTPCommander(ESP32_IP, ESP32_PORT)

    # Start WebSocket thread
    server_loop = asyncio.new_event_loop()
    server_thread = threading.Thread(
        target=start_websocket_server,
        args=(server_loop,),
        daemon=True,
        name="WebSocketThread"
    )
    server_thread.start()

    window_name = "IntelliStream - Low Latency H.264 Tracking + OSNet ReID"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse_click)

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            frame, frame_time = frame_buffer.get(timeout=0.01)

            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif DEBUG_CONFIG["enable_keyboard_shortcuts"] and key in (ord('c'), ord('r')):
                    locked_target_id = None
                elif DEBUG_CONFIG["enable_keyboard_shortcuts"] and ord('1') <= key <= ord('9'):
                    locked_target_id = key - ord('0')
                    locked_target_last_seen = time.time()
                    tracking_mode_label = "MANUAL_OVERRIDE"
                continue

            # Age-Shedding Guardrail: Drop frame if transit + queue latency exceeds budget
            latency_sec = time.time() - frame_time
            if latency_sec > MAX_FRAME_AGE_SEC:
                logger.debug(f"Dropping stale frame (Age: {latency_sec * 1000:.1f} ms)")
                continue

            fps_counter += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                current_fps = fps_counter / (now - fps_start_time)
                fps_counter = 0
                fps_start_time = now

            # YOLO11 + ByteTrack Inference
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_half):
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    device=device,
                    half=False,
                    verbose=False
                )

            locked_box_info = None
            frame_height, frame_width = frame.shape[:2]
            current_person_candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
            active_track_ids: Set[int] = set()

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    coords = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.int().cpu().numpy()
                    track_ids = boxes.id.int().cpu().numpy()

                    for track_id, box, conf, cls_idx in zip(track_ids, coords, confs, classes):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = model.names[int(cls_idx)] if hasattr(model, 'names') else str(cls_idx)
                        active_track_ids.add(int(track_id))

                        if int(cls_idx) == PERSON_CLASS_ID:
                            current_person_candidates.append((int(track_id), (x1, y1, x2, y2)))

                        # Optional debug mouse override
                        if mouse_click_coords is not None:
                            mx, my = mouse_click_coords
                            if x1 <= mx <= x2 and y1 <= my <= y2:
                                locked_target_id = int(track_id)
                                locked_target_last_seen = time.time()
                                tracking_mode_label = "MANUAL_OVERRIDE"

                        if track_id == locked_target_id:
                            locked_box_info = (int(track_id), (x1, y1, x2, y2), class_name, float(conf))
                            locked_target_last_seen = time.time()
                            tracking_mode_label = "BYTETRACK_LOCK"
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                            label = f"ID:{track_id} {class_name} {conf:.2f}"
                            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

            mouse_click_coords = None

            # =========================================================================
            # MODULAR LAYER: HYBRID TRACKING RE-IDENTIFICATION (OSNET)
            # =========================================================================
            if locked_target_id is None or locked_target_id not in active_track_ids:
                if reid_manager.is_registered:
                    tracking_mode_label = "OSNET_SEARCH"
                    match_result = reid_manager.identify_target(
                        frame=frame,
                        candidate_boxes=current_person_candidates,
                        active_track_ids=active_track_ids
                    )
                    if match_result is not None:
                        matched_id, sim_score = match_result
                        locked_target_id = matched_id
                        locked_target_last_seen = time.time()
                        last_similarity_score = sim_score
                        tracking_mode_label = "BYTETRACK_LOCK"
                        
                        for tid, (x1, y1, x2, y2) in current_person_candidates:
                            if tid == locked_target_id:
                                locked_box_info = (tid, (x1, y1, x2, y2), "person", sim_score)
                                break
                else:
                    tracking_mode_label = "AWAITING_REGISTRATION"

            # =========================================================================
            # MODULAR LAYER: ROBOT STEERING LOGIC & DEAD-ZONE CALCULATIONS
            # =========================================================================
            robot_cmd = "S"
            offset_ratio = 0.0
            frame_cx = frame_width // 2

            if locked_box_info is not None:
                tid, (x1, y1, x2, y2), cname, conf = locked_box_info
                target_cx = (x1 + x2) // 2
                target_cy = y2
                
                offset_ratio = (target_cx - frame_cx) / float(frame_width)

                if abs(offset_ratio) <= DEAD_ZONE_RATIO:
                    robot_cmd = "F"
                elif offset_ratio < -DEAD_ZONE_RATIO:
                    robot_cmd = "L"
                else:
                    robot_cmd = "R"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                line_len = max(10, min(x2 - x1, y2 - y1) // 4)
                cv2.line(frame, (x1, y1), (x1 + line_len, y1), (0, 255, 0), 3)
                cv2.line(frame, (x1, y1), (x1, y1 + line_len), (0, 255, 0), 3)
                cv2.line(frame, (x2, y2), (x2 - line_len, y2), (0, 255, 0), 3)
                cv2.line(frame, (x2, y2), (x2, y2 - line_len), (0, 255, 0), 3)

                vector_color = (0, 255, 0) if robot_cmd == "F" else (0, 255, 255)
                cv2.line(frame, (frame_cx, frame_height), (target_cx, target_cy), vector_color, 2, cv2.LINE_AA)
                cv2.circle(frame, (target_cx, target_cy), 5, (0, 0, 255), -1)

                lock_label = f"LOCKED [ID:{tid}] {cname} {conf:.2f}"
                cv2.putText(frame, lock_label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                status_text = f"LOCKED [ID:{tid}] | Conf/Sim: {conf:.2f}"
                status_color = (0, 255, 0)
            else:
                if locked_target_id is not None:
                    time_since_seen = time.time() - locked_target_last_seen
                    if time_since_seen > REID_CONFIG["target_lost_timeout_sec"]:
                        logger.info(f"Target ID:{locked_target_id} lost timeout. Releasing lock.")
                        locked_target_id = None
                        status_text = "UNLOCKED"
                        status_color = (255, 255, 255)
                    else:
                        status_text = f"LOST [ID:{locked_target_id}] ({time_since_seen:.1f}s)"
                        status_color = (0, 165, 255)
                else:
                    status_text = "UNLOCKED"
                    status_color = (255, 255, 255)
                
                robot_cmd = "S"

            robot_commander.send_command(robot_cmd)

            # =========================================================================
            # MODULAR LAYER: ZERO-OVERHEAD ROBOT CONTROL & OPTIONAL DEBUG HUD
            # =========================================================================
            dz_left = int(frame_cx - (frame_width * DEAD_ZONE_RATIO))
            dz_right = int(frame_cx + (frame_width * DEAD_ZONE_RATIO))
            cv2.line(frame, (dz_left, 0), (dz_left, frame_height), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (dz_right, 0), (dz_right, frame_height), (255, 255, 255), 1, cv2.LINE_AA)

            top_bar = frame[0:36, 0:frame_width]
            cv2.rectangle(top_bar, (0, 0), (top_bar.shape[1], 36), (0, 0, 0), -1)

            e2e_latency_ms = (time.time() - frame_time) * 1000.0
            info_line = (f"H.264 @ {current_fps:.1f} FPS | "
                         f"Lat: {e2e_latency_ms:.1f} ms | "
                         f"CMD: [{robot_cmd}] ({offset_ratio:+.2f}) | "
                         f"TARGET: {status_text}")
            cv2.putText(frame, info_line, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

            # Optional Debug HUD Sub-banner (Gated via DEBUG_CONFIG)
            if DEBUG_CONFIG["show_hud_stats"]:
                cache_size = len(reid_manager.embedding_cache)
                max_cache = reid_manager.max_cache_size
                debug_line = (f"MODE: [{tracking_mode_label}] | "
                              f"Active ID: {locked_target_id or 'None'} | "
                              f"Sim: {last_similarity_score:.2f} | "
                              f"Cache: {cache_size}/{max_cache}")
                cv2.putText(frame, debug_line, (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 20), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif DEBUG_CONFIG["enable_keyboard_shortcuts"] and key in (ord('c'), ord('r')):
                locked_target_id = None
                robot_commander.send_command("S")
            elif DEBUG_CONFIG["enable_keyboard_shortcuts"] and ord('1') <= key <= ord('9'):
                locked_target_id = key - ord('0')
                locked_target_last_seen = time.time()
                tracking_mode_label = "MANUAL_OVERRIDE"

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    finally:
        robot_commander.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()