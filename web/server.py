"""
Web Server for Video Classifier Model Demo & Verification Dashboard.
Runs 100% locally and offline on Tornado.
"""
import os
import sys
import json
import time
import base64
import uuid
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import torch
import tornado.ioloop
import tornado.web

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer import ClassifierPipeline
from src.config import ModelConfig, CLASS_NAMES, CHUNKING_POLICIES

DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
UPLOADS_DIR = DATA_DIR / ".cache" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Load metadata for ground-truth lookups
METADATA_PATH = DATA_DIR / "video_metadata.csv"
if METADATA_PATH.exists():
    META_DF = pd.read_csv(METADATA_PATH).drop_duplicates(subset=["video_id"])
    GT_LOOKUP = {row["video_id"]: {
        "chunk_category": row["chunk_category"],
        "sub_category": row.get("sub_category", "Unknown"),
        "domain": row.get("domain", "General")
    } for _, row in META_DF.iterrows()}
else:
    GT_LOOKUP = {}

# User verification feedback tracking in memory
VERIFICATION_LOG: List[Dict[str, Any]] = []

# Initialize pipeline once on server start
print("[Server] Initializing Classifier Pipeline and loading model weights...")
pipeline = ClassifierPipeline()
if pipeline.weights_path.exists():
    pipeline.model.load_state_dict(torch.load(pipeline.weights_path, map_location=pipeline.config.device, weights_only=True))
    pipeline.model.eval()
    print(f"[Server] Successfully loaded weights from {pipeline.weights_path}")
else:
    print("[Server] Warning: Model weights not found. Using initialized model.")


def frame_to_b64(rgb_frame: np.ndarray) -> str:
    """Converts RGB numpy frame to base64 JPEG data URL."""
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(str(STATIC_DIR / "index.html"))


class SamplesHandler(tornado.web.RequestHandler):
    def get(self):
        """Returns list of pre-downloaded sample videos with ground truth."""
        samples = []
        if VIDEOS_DIR.exists():
            for f in sorted(os.listdir(VIDEOS_DIR)):
                if f.endswith(".mp4"):
                    vid_id = f.replace(".mp4", "")
                    file_path = VIDEOS_DIR / f
                    size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
                    meta = GT_LOOKUP.get(vid_id, {
                        "chunk_category": "Unknown",
                        "sub_category": "General",
                        "domain": "General"
                    })
                    samples.append({
                        "video_id": vid_id,
                        "filename": f,
                        "file_size_mb": size_mb,
                        "ground_truth_category": meta["chunk_category"],
                        "sub_category": meta["sub_category"],
                        "domain": meta["domain"]
                    })
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps({"samples": samples, "total": len(samples)}))


class PredictHandler(tornado.web.RequestHandler):
    def post(self):
        """
        Runs inference on either:
        1. A selected sample video (JSON body: {"video_id": "..."} or {"filename": "..."})
        2. An uploaded video file (multipart form data: "video_file")
        """
        target_path = None
        video_filename = None
        is_upload = False
        video_id = None

        if "video_file" in self.request.files:
            file_obj = self.request.files["video_file"][0]
            orig_name = file_obj["filename"]
            unique_name = f"{uuid.uuid4().hex[:8]}_{orig_name}"
            target_path = UPLOADS_DIR / unique_name
            with open(target_path, "wb") as f_out:
                f_out.write(file_obj["body"])
            video_filename = unique_name
            is_upload = True
        else:
            try:
                body = json.loads(self.request.body)
            except Exception:
                body = {}
            requested_id = body.get("video_id") or body.get("filename")
            if requested_id:
                if not requested_id.endswith(".mp4"):
                    requested_id = requested_id + ".mp4"
                target_path = VIDEOS_DIR / requested_id
                video_filename = requested_id
                video_id = requested_id.replace(".mp4", "")

        if not target_path or not target_path.exists():
            self.set_status(400)
            self.write(json.dumps({"error": f"Video file not found or upload missing: {target_path}"}))
            return

        # 1. Feature Extraction & Keyframes
        start_t = time.perf_counter()
        frames, timestamps = pipeline.feature_extractor.sample_keyframes(str(target_path))
        visual_feats = pipeline.feature_extractor.extract_visual_features(frames)
        motion_feats = pipeline.feature_extractor.compute_motion_dynamics(frames)

        # Concatenate visual + motion into temporal sequence
        combined = np.concatenate([visual_feats, motion_feats], axis=-1)
        tensor = torch.from_numpy(combined).unsqueeze(0).to(pipeline.config.device)

        # 2. Model Forward & Calibration
        prediction = pipeline.model.predict(tensor)
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        # 3. Keyframe payloads with base64 thumbnails and attention
        keyframes_payload = []
        attn_weights = prediction.get("attention_weights", [0.0] * len(frames))
        for i, (frame, ts, attn) in enumerate(zip(frames, timestamps, attn_weights)):
            motion_val = float(motion_feats[i][0]) if i < len(motion_feats) else 0.0
            keyframes_payload.append({
                "index": i + 1,
                "timestamp_normalized": round(float(ts), 3),
                "attention_weight": round(float(attn), 4),
                "motion_energy": round(motion_val, 4),
                "thumbnail_b64": frame_to_b64(frame)
            })

        # 4. Ground Truth & Correctness Evaluation
        ground_truth = None
        is_correct = None
        domain = "Custom"
        sub_category = "User Upload"

        if video_id and video_id in GT_LOOKUP:
            ground_truth = GT_LOOKUP[video_id]["chunk_category"]
            sub_category = GT_LOOKUP[video_id]["sub_category"]
            domain = GT_LOOKUP[video_id]["domain"]
            is_correct = (prediction["category"] == ground_truth)

        # Stream URL
        if is_upload:
            stream_url = f"/api/video/uploads/{video_filename}"
        else:
            stream_url = f"/api/video/samples/{video_filename}"

        response_data = {
            "success": True,
            "filename": video_filename,
            "stream_url": stream_url,
            "predicted_category": prediction["category"],
            "confidence": round(prediction["confidence"] * 100, 2),
            "margin": prediction["margin"],
            "is_uncertain": prediction["is_uncertain"],
            "escalate_tier2": prediction["escalate_tier2"],
            "probabilities": {k: round(v * 100, 2) for k, v in prediction["probabilities"].items()},
            "chunking_strategy": prediction["chunking_strategy"],
            "chunking_trigger": prediction["chunking_trigger"],
            "chunking_rationale": prediction["chunking_rationale"],
            "inference_latency_ms": latency_ms,
            "ground_truth": ground_truth,
            "has_ground_truth": ground_truth is not None,
            "is_correct": is_correct,
            "sub_category": sub_category,
            "domain": domain,
            "keyframes": keyframes_payload,
            "categories_list": CLASS_NAMES
        }

        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(response_data))


class VerifyFeedbackHandler(tornado.web.RequestHandler):
    def post(self):
        """Allows user to submit ground truth verdict ('correct' or 'incorrect')."""
        try:
            data = json.loads(self.request.body)
        except Exception:
            data = {}

        verdict = data.get("verdict") # "correct" or "incorrect"
        video_name = data.get("video_name", "unknown")
        pred_cat = data.get("predicted_category")
        user_cat = data.get("user_category") or pred_cat

        if verdict in ["correct", "incorrect"]:
            VERIFICATION_LOG.append({
                "video_name": video_name,
                "predicted": pred_cat,
                "user_selected": user_cat,
                "is_correct": (verdict == "correct"),
                "timestamp": time.time()
            })

        total = len(VERIFICATION_LOG)
        correct_count = sum(1 for item in VERIFICATION_LOG if item["is_correct"])
        acc = round((correct_count / total) * 100, 1) if total > 0 else 0.0

        self.set_header("Content-Type", "application/json")
        self.write(json.dumps({
            "success": True,
            "total_verified": total,
            "correct_count": correct_count,
            "incorrect_count": total - correct_count,
            "user_accuracy_pct": acc
        }))

    def get(self):
        total = len(VERIFICATION_LOG)
        correct_count = sum(1 for item in VERIFICATION_LOG if item["is_correct"])
        acc = round((correct_count / total) * 100, 1) if total > 0 else 0.0
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps({
            "total_verified": total,
            "correct_count": correct_count,
            "incorrect_count": total - correct_count,
            "user_accuracy_pct": acc
        }))


class StatsHandler(tornado.web.RequestHandler):
    def get(self):
        """Returns baseline benchmark metrics and model telemetry."""
        total = len(VERIFICATION_LOG)
        correct_count = sum(1 for item in VERIFICATION_LOG if item["is_correct"])
        acc = round((correct_count / total) * 100, 1) if total > 0 else 0.0

        stats = {
            "benchmark": {
                "top1_accuracy": 60.87,
                "macro_f1": 0.6314,
                "latency_p50_ms": 0.38,
                "latency_p95_ms": 0.63,
                "throughput_fps": 450.27,
                "parameters": 100360,
                "model_size_kb": 392.03,
                "peas_score": 5.565
            },
            "user_verification": {
                "total_verified": total,
                "correct_count": correct_count,
                "incorrect_count": total - correct_count,
                "user_accuracy_pct": acc
            },
            "categories": CLASS_NAMES,
            "offline_status": "100% Local (Zero Internet Dependency)"
        }
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(stats))


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/api/samples", SamplesHandler),
        (r"/api/predict", PredictHandler),
        (r"/api/verify", VerifyFeedbackHandler),
        (r"/api/stats", StatsHandler),
        (r"/api/video/samples/(.*)", tornado.web.StaticFileHandler, {"path": str(VIDEOS_DIR)}),
        (r"/api/video/uploads/(.*)", tornado.web.StaticFileHandler, {"path": str(UPLOADS_DIR)}),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": str(STATIC_DIR)}),
    ], debug=False)


if __name__ == "__main__":
    port = 7860
    app = make_app()
    app.listen(port, address="127.0.0.1")
    print(f"\n=======================================================")
    print(f"  Task 1 Video Classifier Web App is LIVE!")
    print(f"  URL: http://127.0.0.1:{port}")
    print(f"  Mode: 100% Local & Offline")
    print(f"=======================================================\n")
    tornado.ioloop.IOLoop.current().start()
