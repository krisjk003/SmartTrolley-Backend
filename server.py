"""
===============================================================================
ULTRA-LOW LATENCY H.264 WEBSOCKET RECEIVER, YOLO11 TRACKING & ESP32 ROBOT SERVER
WITH INTEGRATED SMART TROLLEY SHOPPING CART BACKEND & OSNET ReID
===============================================================================
Optimized for Low-Latency Android MediaCodec -> PyAV -> YOLO11 ByteTrack Streams
with Integrated ESP32 HTTP Robot Following, Dead-Zone Steering, OSNet ReID,
Non-Blocking Cart Telemetry, ThreadingHTTPServer, and Cart Persistence.
===============================================================================
"""

import asyncio
import base64
import json
import logging
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
import warnings
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Set, Tuple

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
except Exception as e:
    import traceback
    traceback.print_exc()
    FeatureExtractor = None

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SmartTrolleyServer")

# =============================================================================
# TRACKING & STREAM CONFIGURATION
# =============================================================================
HOST = "0.0.0.0"
PORT = 8765
YOLO_MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "bytetrack.yaml"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
MAX_FRAME_AGE_SEC = 0.100  # 100 ms budget to accommodate Wi-Fi jitter
PERSON_CLASS_ID = 0        # COCO class index for 'person'

# =============================================================================
# CENTRALIZED ReID & REGISTRATION CONFIGURATION
# =============================================================================
REID_CONFIG = {
    "model_name": "osnet_x0_5",           # Lightweight model: 0.27 GFLOPs
    "sim_threshold": 0.65,                # Minimum cosine similarity to candidate
    "margin_threshold": 0.15,             # Min similarity gap between 1st and 2nd best candidate
    "search_interval_sec": 0.1,           # Throttle ReID searches to every 100 ms
    "cache_ttl_sec": 2.0,                 # Time-to-live for cached track embeddings
    "max_cache_size": 50,                 # LRU cache upper boundary
    "hysteresis_frames": 2,               # Consecutive positive matches required to lock
    "target_lost_timeout_sec": 5.0,       # Hold lock when occluded before release
    "registration_port": 8080,            # REST endpoint port for Android customer cart
    "distance_metric": "cosine"
}

# =============================================================================
# CART BACKEND TELEMETRY & SESSION CONFIGURATION
# =============================================================================
CART_BACKEND_CONFIG = {
    "trolley_id": "TROLLEY_01",                      # Unique ID for this shopping trolley
    "backend_api_url": "http://127.0.0.1:5000/api",    # Shopping backend base API URL
    "sync_interval_sec": 0.5,                            # Max telemetry broadcast rate (2 Hz)
    "enable_remote_logging": True,                       # Set False to run offline without backend
    "request_timeout_sec": 0.3                           # Fast fail timeout to prevent worker blocking
}

# =============================================================================
# API SECURITY & PAYMENT CONFIGURATION
# =============================================================================
SECURITY_CONFIG = {
    "api_key": "smart-trolley-secret-2026",              # Required header: 'X-API-Key'
    "require_auth_for_cart": True                        # Set True to enforce API Key on cart routes
}

PAYMENT_CONFIG = {
    "upi_id": "smarttrolley.store@okaxis",               # Store UPI VPA
    "payee_name": "SmartTrolley Hypermarket",
    "currency": "INR"
}

SCAN_COOLDOWN_SEC = 1.5                                  # Ignore identical QR scans within 1.5 seconds

DEBUG_CONFIG = {
    "enable_manual_mouse_override": True, # Allow mouse-click locking for testing
    "enable_keyboard_shortcuts": True,    # Allow 'c', 'r', '1'-'9' debug overrides
    "show_hud_stats": True                # Display structured debug metrics on frame HUD
}

# =============================================================================
# ESP32 ROBOT CONTROL CONFIGURATION
# =============================================================================
ESP32_IP = "10.243.255.80"          # Target ESP32 IP Address
ESP32_PORT = 80                      # HTTP Commander Port
DEAD_ZONE_RATIO = 0.15               # +/- 15% from frame center is considered "Straight (/F)"
COMMAND_MIN_INTERVAL = 0.05          # Minimum 50ms between requests (max 20 Hz rate limit)
COMMAND_KEEPALIVE_INTERVAL = 0.5     # Resend command every 500ms if unchanged (watchdog keep-alive)


# =============================================================================
# MODULAR LAYER: SHOPPING CART MANAGER (THREAD-SAFE, VALIDATED & PERSISTED)
# =============================================================================
class CartManager:
    """
    Thread-safe Shopping Cart Manager with JSON catalog validation,
    duplicate scan debouncing, and automatic disk persistence.
    """
    def __init__(
        self,
        products_file_path: str = "products.json",
        carts_file_path: str = "saved_carts.json",
        store_name: str = "SmartTrolley Hypermarket"
    ):
        self.products_file_path = products_file_path
        self.carts_file_path = carts_file_path
        self.store_name = store_name
        self.products: Dict[str, dict] = {}
        self.carts: Dict[str, Dict[str, dict]] = {}  # customer_id -> {product_id: item_data}
        self._last_scanned: Dict[Tuple[str, str], float] = {}  # (customer_id, qr_code) -> timestamp
        self._lock = threading.Lock()
        self._load_products()
        self._load_carts()

    def _load_products(self) -> None:
        """Loads and validates the product catalog once into memory on startup."""
        with self._lock:
            if not os.path.exists(self.products_file_path):
                logger.warning(f"[Cart] Product database '{self.products_file_path}' not found. Initializing empty catalog.")
                self.products = {}
                return
            try:
                with open(self.products_file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                valid_products = {}
                for pid, pdata in raw_data.items():
                    if not isinstance(pdata, dict):
                        logger.warning(f"[Cart] Skipping invalid product ID '{pid}': Entry is not a JSON object.")
                        continue
                    missing_fields = [field for field in ("name", "price", "category") if field not in pdata]
                    if missing_fields:
                        logger.warning(f"[Cart] Skipping invalid product ID '{pid}': Missing fields {missing_fields}.")
                        continue
                    valid_products[pid] = {
                        "name": str(pdata["name"]),
                        "price": float(pdata["price"]),
                        "category": str(pdata["category"])
                    }

                self.products = valid_products
                logger.info(f"[Cart] Successfully validated and loaded {len(self.products)} products from '{self.products_file_path}'.")
            except Exception as e:
                logger.error(f"[Cart] Failed to load '{self.products_file_path}': {e}")
                self.products = {}

    def _load_carts(self) -> None:
        """Restores persisted customer carts from disk on startup."""
        with self._lock:
            if not os.path.exists(self.carts_file_path):
                logger.info(f"[Cart] No persisted carts file found at '{self.carts_file_path}'. Starting clean.")
                self.carts = {}
                return
            try:
                with open(self.carts_file_path, "r", encoding="utf-8") as f:
                    self.carts = json.load(f)
                logger.info(f"[Cart] Restored {len(self.carts)} customer cart(s) from '{self.carts_file_path}'.")
            except Exception as e:
                logger.warning(f"[Cart] Could not restore carts from '{self.carts_file_path}': {e}. Starting clean.")
                self.carts = {}

    def _save_carts(self) -> None:
        """Persists customer carts to disk (must be called with _lock acquired)."""
        try:
            with open(self.carts_file_path, "w", encoding="utf-8") as f:
                json.dump(self.carts, f, indent=2)
            logger.debug(f"[Cart] Successfully persisted carts to '{self.carts_file_path}'.")
        except Exception as e:
            logger.error(f"[Cart] Error saving cart persistence to '{self.carts_file_path}': {e}")

    def _get_cart_total(self, customer_id: str) -> float:
        """Internal helper to calculate grand total (must be called with _lock acquired)."""
        cart = self.carts.get(customer_id, {})
        return sum(item["subtotal"] for item in cart.values())

    def scan_item(self, customer_id: str, qr_code: str, trolley_id: str = "TROLLEY_01") -> Tuple[bool, int, dict]:
        """
        Validates QR code, enforces scan debouncing, and adds/increments product in cart.
        Returns: (success: bool, status_code: int, response_dict: dict)
        """
        with self._lock:
            now = time.time()
            last_scan_time = self._last_scanned.get((customer_id, qr_code), 0.0)

            # Duplicate QR scan protection
            if now - last_scan_time < SCAN_COOLDOWN_SEC:
                logger.warning(f"[Cart] Duplicate Scan Rejection -> Customer: '{customer_id}' scanned '{qr_code}' within {SCAN_COOLDOWN_SEC}s.")
                return False, 429, {
                    "success": False,
                    "error": "DUPLICATE_SCAN",
                    "message": "Duplicate QR scan ignored. Please wait before scanning the same item again."
                }

            if qr_code not in self.products:
                logger.warning(f"[Cart] Scan Failed -> Unknown QR '{qr_code}' for customer '{customer_id}'")
                return False, 404, {
                    "success": False,
                    "error": "PRODUCT_NOT_FOUND",
                    "message": f"Product QR '{qr_code}' not found in catalog."
                }

            self._last_scanned[(customer_id, qr_code)] = now

            product_info = self.products[qr_code]
            if customer_id not in self.carts:
                self.carts[customer_id] = {}

            customer_cart = self.carts[customer_id]

            if qr_code in customer_cart:
                customer_cart[qr_code]["qty"] += 1
            else:
                customer_cart[qr_code] = {
                    "id": qr_code,
                    "name": product_info["name"],
                    "price": product_info["price"],
                    "qty": 1,
                    "subtotal": 0.0
                }

            item = customer_cart[qr_code]
            item["subtotal"] = round(item["price"] * item["qty"], 2)
            cart_total = round(self._get_cart_total(customer_id), 2)

            self._save_carts()
            logger.info(f"[Cart] Scan Success -> '{item['name']}' (Qty: {item['qty']}) | Customer: '{customer_id}' | Total: {cart_total}")

            return True, 200, {
                "success": True,
                "product": item["name"],
                "quantity": item["qty"],
                "unit_price": item["price"],
                "subtotal": item["subtotal"],
                "cart_total": cart_total
            }

    def get_cart(self, customer_id: str) -> Tuple[int, dict]:
        """Retrieves formatted item list and grand total for a customer."""
        with self._lock:
            cart = self.carts.get(customer_id, {})
            items_list = list(cart.values())
            grand_total = round(self._get_cart_total(customer_id), 2)
            return 200, {
                "items": items_list,
                "grand_total": grand_total
            }

    def remove_item(self, customer_id: str, qr_code: str) -> Tuple[bool, int, dict]:
        """Decrements product quantity by 1. Removes item completely if quantity reaches 0."""
        with self._lock:
            cart = self.carts.get(customer_id, {})
            if qr_code not in cart:
                return False, 404, {
                    "success": False,
                    "error": "ITEM_NOT_IN_CART",
                    "message": f"Product '{qr_code}' not found in cart."
                }

            item = cart[qr_code]
            item["qty"] -= 1

            if item["qty"] <= 0:
                removed_name = item["name"]
                del cart[qr_code]
                logger.info(f"[Cart] Item Removed -> '{removed_name}' | Customer: '{customer_id}'")
            else:
                item["subtotal"] = round(item["price"] * item["qty"], 2)
                logger.info(f"[Cart] Item Decremented -> '{item['name']}' (New Qty: {item['qty']}) | Customer: '{customer_id}'")

            cart_total = round(self._get_cart_total(customer_id), 2)
            self._save_carts()

            return True, 200, {
                "success": True,
                "cart_total": cart_total
            }

    def clear_cart(self, customer_id: str) -> Tuple[int, dict]:
        """Clears the customer's cart and saves to disk."""
        with self._lock:
            if customer_id in self.carts:
                self.carts[customer_id] = {}
            self._save_carts()
            logger.info(f"[Cart] Cart Cleared for Customer: '{customer_id}'")
            return 200, {"success": True, "message": f"Cart cleared for customer '{customer_id}'."}

    def checkout(self, customer_id: str) -> Tuple[int, dict]:
        """Generates itemized receipt data and UPI payment QR deep-link string."""
        with self._lock:
            cart = self.carts.get(customer_id, {})
            items_list = list(cart.values())
            grand_total = round(self._get_cart_total(customer_id), 2)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Generate dynamic UPI QR string
            payee_encoded = urllib.parse.quote(PAYMENT_CONFIG["payee_name"])
            upi_qr_string = (
                f"upi://pay?pa={PAYMENT_CONFIG['upi_id']}"
                f"&pn={payee_encoded}"
                f"&am={grand_total}"
                f"&cu={PAYMENT_CONFIG['currency']}"
            )

            logger.info(f"[Cart] Checkout Generated for Customer: '{customer_id}' | Grand Total: {grand_total}")

            return 200, {
                "store_name": self.store_name,
                "customer_id": customer_id,
                "time": timestamp,
                "items": items_list,
                "grand_total": grand_total,
                "payment_info": {
                    "upi_id": PAYMENT_CONFIG["upi_id"],
                    "payee_name": PAYMENT_CONFIG["payee_name"],
                    "currency": PAYMENT_CONFIG["currency"],
                    "upi_qr_string": upi_qr_string
                }
            }


# =============================================================================
# MODULAR LAYER: CART BACKEND & SESSION TELEMETRY REPORTER
# =============================================================================
class CartBackendReporter:
    """
    Non-blocking background worker that broadcasts trolley tracking state, shopper
    session events, and system telemetry to your main shopping backend API.
    """
    def __init__(self, config: dict):
        self.config = config
        self.trolley_id = config.get("trolley_id", "TROLLEY_01")
        self.base_url = config.get("backend_api_url", "").rstrip("/")
        self.sync_interval = config.get("sync_interval_sec", 0.5)
        self.timeout = config.get("request_timeout_sec", 0.3)
        self.enabled = config.get("enable_remote_logging", True)

        self._state_lock = threading.Lock()
        self._current_state = {
            "trolley_id": self.trolley_id,
            "customer_id": None,
            "tracking_mode": "OFFLINE",
            "locked_target_id": None,
            "robot_cmd": "S",
            "similarity": 0.0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "timestamp": time.time()
        }

        self._event_queue: queue.Queue = queue.Queue(maxsize=30)
        self._running = True

        if self.enabled and self.base_url:
            self.thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="CartBackendReporterThread"
            )
            self.thread.start()
            logger.info(f"Cart Backend Reporter initialized -> {self.base_url}")
        else:
            logger.info("Cart Backend Reporter is disabled or missing base_url.")

    def update_state(
        self,
        customer_id: Optional[str],
        tracking_mode: str,
        locked_target_id: Optional[int],
        robot_cmd: str,
        similarity: float,
        fps: float,
        latency_ms: float
    ) -> None:
        """Thread-safe update of the latest trolley operational state."""
        with self._state_lock:
            self._current_state.update({
                "customer_id": customer_id,
                "tracking_mode": tracking_mode,
                "locked_target_id": locked_target_id,
                "robot_cmd": robot_cmd,
                "similarity": round(float(similarity), 3),
                "fps": round(float(fps), 1),
                "latency_ms": round(float(latency_ms), 1),
                "timestamp": time.time()
            })

    def send_event(self, event_type: str, details: Optional[dict] = None) -> None:
        """Queue a priority event (e.g. CUSTOMER_REGISTERED, TARGET_LOCKED, TARGET_LOST)."""
        if not self.enabled:
            return
        payload = {
            "trolley_id": self.trolley_id,
            "event": event_type,
            "timestamp": time.time(),
            "details": details or {}
        }
        try:
            self._event_queue.put_nowait(payload)
        except queue.Full:
            logger.warning(f"Backend event queue full. Dropped event: {event_type}")

    def _post_json(self, endpoint: str, payload: dict) -> None:
        if not self.base_url:
            return
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                _ = resp.read()
        except Exception as e:
            logger.debug(f"Cart backend request to '{endpoint}' failed: {e}")

    def _worker_loop(self) -> None:
        last_sync_time = 0.0
        while self._running:
            # 1. Flush priority events immediately
            while not self._event_queue.empty():
                try:
                    event_payload = self._event_queue.get_nowait()
                    self._post_json("trolley/event", event_payload)
                except queue.Empty:
                    break

            # 2. Broadcast periodic telemetry state
            now = time.time()
            if now - last_sync_time >= self.sync_interval:
                with self._state_lock:
                    state_snapshot = dict(self._current_state)
                self._post_json("trolley/status", state_snapshot)
                last_sync_time = now

            time.sleep(0.05)

    def get_current_state(self) -> dict:
        with self._state_lock:
            return dict(self._current_state)

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.5)


# =============================================================================
# MODULAR LAYER: PERSON RE-IDENTIFICATION (OSNET) MANAGER
# =============================================================================
class PersonReIDManager:
    """
    Production Person Re-Identification using OSNet (TorchReID).
    """
    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.sim_threshold = config.get("sim_threshold", 0.55)
        self.margin_threshold = config.get("margin_threshold", 0.15)
        self.search_interval = config.get("search_interval_sec", 0.15)
        self.cache_ttl = config.get("cache_ttl_sec", 2.0)
        self.max_cache_size = config.get("max_cache_size", 50)
        self.hysteresis_target = config.get("hysteresis_frames", 2)

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

        # ---------------------------------------------------------------------
        # AUTOMATIC LOADING OF PREVIOUSLY SAVED EMBEDDINGS
        # ---------------------------------------------------------------------
        reg_dir = "registered_customers"
        if os.path.exists(reg_dir) and os.path.isdir(reg_dir):
            loaded_count = 0
            for fname in os.listdir(reg_dir):
                if fname.endswith(".pt"):
                    customer_id = os.path.splitext(fname)[0]
                    file_path = os.path.join(reg_dir, fname)
                    try:
                        embedding = torch.load(file_path, map_location=self.device)
                        self.customers[customer_id] = embedding
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load customer embedding '{fname}': {e}")
            
            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} customer embedding(s) from '{reg_dir}'.")
                if len(self.customers) == 1:
                    single_id = list(self.customers.keys())[0]
                    self.active_customer_id = single_id
                    self.is_registered = True
                    logger.info(f"Exactly one customer found. Automatically activated customer '{single_id}'.")
            else:
                logger.info(f"No .pt embedding files found in '{reg_dir}'.")
        else:
            logger.info(f"Registration directory '{reg_dir}' does not exist. Starting with empty customer registry.")

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
        Validates reference images, detects human ROIs via YOLO11, generates embeddings,
        averages them into a single vector, and registers the customer.
        """
        if self.extractor is None:
            return False, "OSNet feature extractor is not initialized."

        if len(bgr_images) < 2:
            msg = "Registration requires at least 2 reference images."
            logger.error(msg)
            return False, msg

        raw_embeddings = []
        logger.info(f"Processing {len(bgr_images)} reference images for customer '{customer_id}'...")

        for idx, img in enumerate(bgr_images, 1):
            if img is None or img.size == 0:
                msg = f"Reference image #{idx} is empty or corrupted."
                logger.error(msg)
                return False, msg

            # Detect ONLY person class (0) in reference photo
            results = yolo_model.predict(source=img, conf=0.40, classes=[PERSON_CLASS_ID], verbose=False)
            best_crop = None
            max_area = 0

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes.xyxy.cpu().numpy():
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
        os.makedirs("registered_customers", exist_ok=True)
        torch.save(
            unified_embedding.cpu(),
            f"registered_customers/{customer_id}.pt"
        )
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
            stale_keys = [
                tid for tid, (_, ts) in self.embedding_cache.items()
                if (tid not in active_track_ids) or (now - ts > self.cache_ttl)
            ]
            for key in stale_keys:
                del self.embedding_cache[key]

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
        active customer's unified reference embedding, enforced by a Margin Check.
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
        second_highest_sim = -1.0
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
            
            # Maintain Top-1 and Top-2 similarities for Margin Check
            if sim > highest_sim:
                second_highest_sim = highest_sim
                highest_sim = sim
                best_match_id = track_id
            elif sim > second_highest_sim:
                second_highest_sim = sim

        if second_highest_sim < 0:
            margin = float("inf")
        else:
            margin = highest_sim - second_highest_sim

        if best_match_id is not None and highest_sim >= self.sim_threshold:
            # Enforce Margin Check: Must lead second-best by at least margin_threshold
            if margin < self.margin_threshold:
                logger.info(
                    f"ReID Ambiguity -> ID={best_match_id} (Sim={highest_sim:.2f}) rejected: "
                    f"Margin to 2nd best ({second_highest_sim:.2f}) is {margin:.2f} < {self.margin_threshold}"
                )
                self.reset_hysteresis()
                return None

            logger.info(
              f"Best Match: ID={best_match_id}, Sim={highest_sim:.3f}, Margin={margin:.3f}"
            )
            if best_match_id == self.hysteresis_candidate_id:
                self.hysteresis_count += 1
            else:
                self.hysteresis_candidate_id = best_match_id
                self.hysteresis_count = 1

            if self.hysteresis_count >= self.hysteresis_target:
                logger.info(f"OSNet ReID Match Confirmed -> Lock ID:{best_match_id} (Sim: {highest_sim:.2f}, Margin: {margin:.2f})")
                self.reset_hysteresis()
                return best_match_id, highest_sim
        else:
            self.reset_hysteresis()

        return None


# =============================================================================
# MODULAR LAYER: ANDROID HTTP REGISTRATION, STATUS & SHOPPING CART REST SERVER
# =============================================================================
class SmartTrolleyHTTPHandler(BaseHTTPRequestHandler):
    """
    Unified HTTP REST Handler serving real-time tracking status and managing
    all shopping cart endpoints (/scan, /cart, /remove, /clear, /checkout).
    Note: Customer Registration is now handled via WebSocket text frames.
    """
    reid_manager: Optional[PersonReIDManager] = None
    yolo_model: Optional[YOLO] = None
    cart_reporter: Optional[CartBackendReporter] = None
    cart_manager: Optional[CartManager] = None

    def _send_json_response(self, status_code: int, payload: dict) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _read_json_payload(self) -> Optional[dict]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return None
        try:
            post_data = self.rfile.read(content_length)
            return json.loads(post_data.decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse JSON body: {e}")
            return None

    def _is_authenticated(self, query_params: dict) -> bool:
        """Validates API Key from HTTP Header ('X-API-Key') or query param ('api_key')."""
        if not SECURITY_CONFIG.get("require_auth_for_cart", True):
            return True
        header_key = self.headers.get("X-API-Key")
        query_key = query_params.get("api_key", [None])[0]
        valid_key = SECURITY_CONFIG["api_key"]
        return header_key == valid_key or query_key == valid_key

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # AUTHENTICATION GUARDRAIL FOR CART ROUTING
        if not self._is_authenticated(query_params):
            logger.warning(f"[Auth] Unauthorized POST attempt from {self.client_address[0]} to '{path}'.")
            self._send_json_response(401, {
                "success": False,
                "error": "UNAUTHORIZED",
                "message": "Missing or invalid X-API-Key HTTP header."
            })
            return

        # 1. QR SCAN ITEM ENDPOINT
        if path == "/scan":
            payload = self._read_json_payload()
            if not payload:
                self._send_json_response(400, {
                    "success": False,
                    "error": "BAD_REQUEST",
                    "message": "Invalid JSON body."
                })
                return

            customer_id = payload.get("customer_id")
            trolley_id = payload.get("trolley_id")
            qr_code = payload.get("qr")

            if not customer_id or not trolley_id or not qr_code:
                self._send_json_response(400, {
                    "success": False,
                    "error": "MISSING_FIELDS",
                    "message": "Missing required fields: 'customer_id', 'trolley_id', and 'qr' are mandatory."
                })
                return

            _, status_code, result = self.cart_manager.scan_item(
                customer_id=str(customer_id),
                qr_code=str(qr_code),
                trolley_id=str(trolley_id)
            )
            self._send_json_response(status_code, result)
            return

        # 2. REMOVE PRODUCT ENDPOINT
        elif path == "/remove":
            payload = self._read_json_payload()
            if not payload:
                self._send_json_response(400, {
                    "success": False,
                    "error": "BAD_REQUEST",
                    "message": "Invalid JSON body."
                })
                return

            customer_id = payload.get("customer_id")
            qr_code = payload.get("qr")

            if not customer_id or not qr_code:
                self._send_json_response(400, {
                    "success": False,
                    "error": "MISSING_FIELDS",
                    "message": "Missing required fields: 'customer_id' and 'qr' are mandatory."
                })
                return

            _, status_code, result = self.cart_manager.remove_item(
                customer_id=str(customer_id),
                qr_code=str(qr_code)
            )
            self._send_json_response(status_code, result)
            return

        # 3. CLEAR CART ENDPOINT
        elif path == "/clear":
            payload = self._read_json_payload()
            if not payload:
                self._send_json_response(400, {
                    "success": False,
                    "error": "BAD_REQUEST",
                    "message": "Invalid JSON body."
                })
                return

            customer_id = payload.get("customer_id")
            if not customer_id:
                self._send_json_response(400, {
                    "success": False,
                    "error": "MISSING_FIELDS",
                    "message": "Missing required field: 'customer_id'."
                })
                return

            status_code, result = self.cart_manager.clear_cart(customer_id=str(customer_id))
            self._send_json_response(status_code, result)
            return

        # FALLBACK 404
        self._send_json_response(404, {
            "success": False,
            "error": "NOT_FOUND",
            "message": f"Endpoint '{path}' not found."
        })

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # 1. REAL-TIME TRACKING STATUS ENDPOINT
        if path == "/status":
            state_data = self.cart_reporter.get_current_state() if self.cart_reporter else {}
            self._send_json_response(200, state_data)
            return

        # AUTHENTICATION GUARDRAIL FOR CART ROUTING
        if not self._is_authenticated(query_params):
            logger.warning(f"[Auth] Unauthorized GET attempt from {self.client_address[0]} to '{path}'.")
            self._send_json_response(401, {
                "success": False,
                "error": "UNAUTHORIZED",
                "message": "Missing or invalid X-API-Key header or query parameter."
            })
            return

        # 2. GET SHOPPING CART ENDPOINT
        if path == "/cart":
            customer_id = query_params.get("customer_id", ["default"])[0]
            status_code, result = self.cart_manager.get_cart(customer_id=customer_id)
            self._send_json_response(status_code, result)
            return

        # 3. GET CHECKOUT INVOICE ENDPOINT
        elif path == "/checkout":
            customer_id = query_params.get("customer_id", ["default"])[0]
            status_code, result = self.cart_manager.checkout(customer_id=customer_id)
            self._send_json_response(status_code, result)
            return

        # FALLBACK 404
        self._send_json_response(404, {
            "success": False,
            "error": "NOT_FOUND",
            "message": f"Endpoint '{path}' not found."
        })

    def log_message(self, format, *args):
        pass  # Suppress default HTTP console spam


def start_http_server(
    reid_manager: PersonReIDManager,
    yolo_model: YOLO,
    cart_reporter: CartBackendReporter,
    cart_manager: CartManager,
    port: int
):
    """Launches the ThreadingHTTPServer in a daemon thread for concurrent Android request handling."""
    SmartTrolleyHTTPHandler.reid_manager = reid_manager
    SmartTrolleyHTTPHandler.yolo_model = yolo_model
    SmartTrolleyHTTPHandler.cart_reporter = cart_reporter
    SmartTrolleyHTTPHandler.cart_manager = cart_manager
    server_address = ('0.0.0.0', port)
    httpd = ThreadingHTTPServer(server_address, SmartTrolleyHTTPHandler)
    logger.info(f"SmartTrolley Threaded HTTP Server listening on http://0.0.0.0:{port} (/scan, /cart, /checkout, etc.)")
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


async def websocket_handler(
    websocket,
    path=None,
    reid_manager: Optional[PersonReIDManager] = None,
    yolo_model: Optional[YOLO] = None,
    cart_reporter: Optional[CartBackendReporter] = None
):
    """
    Bidirectional WebSocket Handler:
    - Raw binary bytes -> Routed to low-latency PyAV H.264 decoder.
    - JSON text strings -> Routed to non-blocking OSNet customer registration.
    """
    client_address = websocket.remote_address
    logger.info(f"Android client connected via WebSocket: {client_address}")
    decoder = LowLatencyH264Decoder()

    try:
        async for message in websocket:
            # 1. BINARY FRAMES: LOW-LATENCY H.264 VIDEO STREAM
            if isinstance(message, bytes):
                recv_time = time.time()
                frames = decoder.decode_packet(message)
                for frame in frames:
                    frame_buffer.put(frame, recv_time)

            # 2. TEXT FRAMES: JSON COMMANDS (E.G., CUSTOMER REGISTRATION VIA IMAGES)
            elif isinstance(message, str):
                try:
                    payload = json.loads(message)
                    action = payload.get("action")

                    if action == "register":
                        customer_id = str(payload.get("customer_id", "default"))
                        images_b64 = payload.get("images", [])

                        if not reid_manager or not yolo_model:
                            await websocket.send(json.dumps({
                                "action": "register_response",
                                "success": False,
                                "error": "SERVER_UNINITIALIZED",
                                "message": "ReID or YOLO model not bound to WebSocket handler."
                            }))
                            continue

                        # Decode Base64 images into OpenCV BGR arrays
                        bgr_images = []
                        for b64_str in images_b64:
                            try:
                                img_bytes = base64.b64decode(b64_str)
                                np_arr = np.frombuffer(img_bytes, np.uint8)
                                bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                if bgr_img is not None:
                                    bgr_images.append(bgr_img)
                            except Exception as e:
                                logger.warning(f"Failed to decode a Base64 registration frame: {e}")

                        if len(bgr_images) < 2:
                            await websocket.send(json.dumps({
                                "action": "register_response",
                                "success": False,
                                "error": "INSUFFICIENT_IMAGES",
                                "message": "Registration requires at least 2 valid reference images."
                            }))
                            continue

                        logger.info(f"[WebSocket] Starting async OSNet registration for '{customer_id}' ({len(bgr_images)} frames)...")

                        # Run CUDA-heavy OSNet embedding extraction in background thread
                        success, result_msg = await asyncio.to_thread(
                            reid_manager.register_customer_from_images,
                            yolo_model=yolo_model,
                            bgr_images=bgr_images,
                            customer_id=customer_id
                        )

                        if success and cart_reporter:
                            cart_reporter.send_event("CUSTOMER_REGISTERED", {
                                "customer_id": customer_id,
                                "reference_count": len(bgr_images),
                                "transport": "websocket"
                            })

                        # Send JSON confirmation back to Android over WebSocket
                        await websocket.send(json.dumps({
                            "action": "register_response",
                            "success": success,
                            "customer_id": customer_id,
                            "error": None if success else "REGISTRATION_FAILED",
                            "message": result_msg
                        }))

                    else:
                        logger.warning(f"[WebSocket] Unknown command action received: {action}")
                        await websocket.send(json.dumps({
                            "action": "error",
                            "success": False,
                            "message": f"Unknown action: '{action}'"
                        }))

                except json.JSONDecodeError:
                    logger.error("[WebSocket] Malformed JSON text frame received.")
                    await websocket.send(json.dumps({
                        "action": "error",
                        "success": False,
                        "message": "Malformed JSON payload."
                    }))
                except Exception as e:
                    logger.error(f"[WebSocket] Unexpected error during registration processing: {e}")
                    await websocket.send(json.dumps({
                        "action": "error",
                        "success": False,
                        "message": str(e)
                    }))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Android client disconnected: {client_address}")
    finally:
        decoder.reset()


def start_websocket_server(
    loop: asyncio.AbstractEventLoop,
    reid_manager: PersonReIDManager,
    yolo_model: YOLO,
    cart_reporter: CartBackendReporter
):
    """Runs WebSocket server loop in background daemon thread with ReID & YOLO context."""
    asyncio.set_event_loop(loop)
    
    # Create closure handler to inject dependencies into websocket_handler
    async def bound_handler(ws, path=None):
        await websocket_handler(
            ws,
            path=path,
            reid_manager=reid_manager,
            yolo_model=yolo_model,
            cart_reporter=cart_reporter
        )

    server = websockets.serve(
        bound_handler,
        HOST,
        PORT,
        max_queue=1,
        max_size=None,          # Unlimited frame size for large Base64 arrays
        ping_interval=20,
        ping_timeout=10
    )
    logger.info(f"Bidirectional WebSocket server listening on ws://{HOST}:{PORT}")
    loop.run_until_complete(server)
    loop.run_forever()


# =============================================================================
# MAIN EXECUTABLE & HYBRID TRACKING LOOP
# =============================================================================
def main():
    
    global locked_target_id, mouse_click_coords, locked_target_last_seen
    global last_similarity_score, tracking_mode_label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = True

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
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_half):
            model.track(
                source=dummy_input,
                persist=True,
                tracker=TRACKER_CONFIG,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=[PERSON_CLASS_ID], # ONLY track humans
                device=device,
                half=use_half,
                verbose=False
            )

    # -------------------------------------------------------------------------
    # INITIALIZE CART MANAGER, TELEMETRY REPORTER, ReID MANAGER & REST SERVER
    # -------------------------------------------------------------------------
    cart_manager = CartManager(
        products_file_path="products.json",
        carts_file_path="saved_carts.json"
    )
    cart_reporter = CartBackendReporter(config=CART_BACKEND_CONFIG)
    reid_manager = PersonReIDManager(config=REID_CONFIG, device=device)

    reg_thread = threading.Thread(
        target=start_http_server,
        args=(reid_manager, model, cart_reporter, cart_manager, REID_CONFIG["registration_port"]),
        daemon=True,
        name="AndroidHTTPRestThread"
    )
    reg_thread.start()

    # Initialize Background ESP32 Robot Commander
    robot_commander = ESP32HTTPCommander(ESP32_IP, ESP32_PORT)
  

    # Start Bidirectional WebSocket thread (Video + Image Registration)
    server_loop = asyncio.new_event_loop()
    server_thread = threading.Thread(
        target=start_websocket_server,
        args=(server_loop, reid_manager, model, cart_reporter),
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
                    robot_commander.send_command("S")
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

            # =========================================================================
            # YOLO11 + ByteTrack Inference (STRICTLY HUMANS ONLY via classes=[0])
            # =========================================================================
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_half):
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    classes=[PERSON_CLASS_ID],  # ENFORCES HUMAN-ONLY DETECTIONS
                    device=device,
                    half=use_half,
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
                        active_track_ids.add(int(track_id))
                        current_person_candidates.append((int(track_id), (x1, y1, x2, y2)))

                        # Optional debug mouse override
                        if mouse_click_coords is not None:
                            mx, my = mouse_click_coords
                            if x1 <= mx <= x2 and y1 <= my <= y2:
                                locked_target_id = int(track_id)
                                locked_target_last_seen = time.time()
                                tracking_mode_label = "MANUAL_OVERRIDE"

                        if track_id == locked_target_id:
                            locked_box_info = (int(track_id), (x1, y1, x2, y2), "person", float(conf))
                            locked_target_last_seen = time.time()
                            tracking_mode_label = "BYTETRACK_LOCK"
                        else:
                            # Draw ONLY human candidates (subtle cyan box for non-locked people)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                            label = f"ID:{track_id} person {conf:.2f}"
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
                        
                        cart_reporter.send_event("TARGET_LOCKED", {
                            "track_id": locked_target_id,
                            "similarity": sim_score,
                            "customer_id": reid_manager.active_customer_id
                        })
                        
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

                # Shrink only the displayed lock box
                shrink = 0.50

                w = x2 - x1
                h = y2 - y1

                dx = int(w * shrink / 2)
                dy = int(h * shrink / 2)

                x1 += dx
                y1 += dy
                x2 -= dx
                y2 -= dy

                target_cx = (x1 + x2) // 2
                target_cy = y2
                
                offset_ratio = (target_cx - frame_cx) / float(frame_width)

                if abs(offset_ratio) <= DEAD_ZONE_RATIO:
                    robot_cmd = "F"
                elif offset_ratio < -DEAD_ZONE_RATIO:
                    robot_cmd = "L"
                else:
                    robot_cmd = "R"

                # Render prominent TARGET visual highlights around the matched person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                line_len = max(10, min(x2 - x1, y2 - y1) // 4)
                cv2.line(frame, (x1, y1), (x1 + line_len, y1), (0, 255, 0), 3)
                cv2.line(frame, (x1, y1), (x1, y1 + line_len), (0, 255, 0), 3)
                cv2.line(frame, (x2, y2), (x2 - line_len, y2), (0, 255, 0), 3)
                cv2.line(frame, (x2, y2), (x2, y2 - line_len), (0, 255, 0), 3)

                vector_color = (0, 255, 0) if robot_cmd == "F" else (0, 255, 255)
                cv2.line(frame, (frame_cx, frame_height), (target_cx, target_cy), vector_color, 2, cv2.LINE_AA)
                cv2.circle(frame, (target_cx, target_cy), 5, (0, 0, 255), -1)

                lock_label = f"LOCKED TARGET [ID:{tid}] {cname} {conf:.2f}"
                cv2.putText(frame, lock_label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                status_text = f"LOCKED [ID:{tid}] | Conf/Sim: {conf:.2f}"
                status_color = (0, 255, 0)
            else:
                if locked_target_id is not None:
                    time_since_seen = time.time() - locked_target_last_seen
                    if time_since_seen > REID_CONFIG["target_lost_timeout_sec"]:
                        logger.info(f"Target ID:{locked_target_id} lost timeout. Releasing lock.")
                        cart_reporter.send_event("TARGET_LOST", {
                            "last_track_id": locked_target_id,
                            "customer_id": reid_manager.active_customer_id
                        })
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
            print(f"command is {robot_cmd}")

            # =========================================================================
            # NON-BLOCKING CART TELEMETRY BROADCAST
            # =========================================================================
            e2e_latency_ms = (time.time() - frame_time) * 1000.0
            cart_reporter.update_state(
                customer_id=reid_manager.active_customer_id if reid_manager.is_registered else None,
                tracking_mode=tracking_mode_label,
                locked_target_id=locked_target_id,
                robot_cmd=robot_cmd,
                similarity=last_similarity_score,
                fps=current_fps,
                latency_ms=e2e_latency_ms
            )

            # =========================================================================
            # MODULAR LAYER: ZERO-OVERHEAD ROBOT CONTROL & OPTIONAL DEBUG HUD
            # =========================================================================
            dz_left = int(frame_cx - (frame_width * DEAD_ZONE_RATIO))
            dz_right = int(frame_cx + (frame_width * DEAD_ZONE_RATIO))
            cv2.line(frame, (dz_left, 0), (dz_left, frame_height), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (dz_right, 0), (dz_right, frame_height), (255, 255, 255), 1, cv2.LINE_AA)

            top_bar = frame[0:36, 0:frame_width]
            cv2.rectangle(top_bar, (0, 0), (top_bar.shape[1], 36), (0, 0, 0), -1)

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
        cart_reporter.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()