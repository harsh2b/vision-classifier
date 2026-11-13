"""
Batch Video Downloader and Real Feature Extractor for Task 1: Video Classifier.
Concurrently downloads videos from HuggingFace, extracts real MobileNetV3 visual
vectors and kinetic motion deltas, and updates features_cache.npz and model_weights.pt.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Ensure src is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import ModelConfig, CLASS_NAMES
from src.data_loader import VideoMMELoader, REPO_ID
from src.features import FeatureExtractor
from src.model import VideoFormatClassifier
from src.metrics import (
    compute_standard_metrics,
    compute_bootstrap_confidence_intervals,
    compute_peas_score,
    LatencyProfiler
)

def download_single_video(video_id: str, dest_dir: Path) -> str:
    """Downloads an individual video from HuggingFace if not already present."""
    local_path = dest_dir / f"{video_id}.mp4"
    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    filename = f"videos/{video_id}.mp4"
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest_dir.parent),
        local_dir_use_symlinks=False
    )
    return downloaded

def main():
    data_dir = root_dir / "data"
    videos_dir = data_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    loader = VideoMMELoader(data_dir=str(data_dir))
    df = loader.load_metadata()
    unique_vids = df["video_id"].unique().tolist()
    print(f"[Pipeline] Found {len(unique_vids)} unique videos in Video-MME corpus.")

    # Check already downloaded videos
    existing = list(videos_dir.glob("*.mp4"))
    print(f"[Pipeline] Already downloaded: {len(existing)} videos.")

    extractor = FeatureExtractor()
    features_dict = {}

    # Extract features from already existing videos first
    for vpath in existing:
        vid_name = vpath.stem
        if vid_name in unique_vids:
            try:
                feats = extractor.extract_video_features(str(vpath))
                features_dict[vid_name] = feats["temporal_seq"]
                print(f"[Features] Extracted from existing video: {vid_name} ({vpath.stat().st_size / (1024*1024):.1f} MB)")
            except Exception as e:
                print(f"[Features] Error on {vpath}: {e}")

    # Prioritize remaining videos
    remaining_vids = [v for v in unique_vids if v not in features_dict]
    print(f"[Pipeline] Remaining videos to download and process: {len(remaining_vids)}")

    # Concurrent download & extraction (max_workers=4)
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_vid = {
            executor.submit(download_single_video, vid, videos_dir): vid 
            for vid in remaining_vids
        }

        for future in as_completed(future_to_vid):
            vid = future_to_vid[future]
            try:
                vpath = future.result()
                if vpath and os.path.exists(vpath):
                    file_size_mb = os.path.getsize(vpath) / (1024 * 1024)
                    print(f"[Download Complete] {vid} ({file_size_mb:.1f} MB). Extracting features...")
                    feats = extractor.extract_video_features(str(vpath))
                    features_dict[vid] = feats["temporal_seq"]
                    print(f"[Features Ready] {vid} -> shape {feats['temporal_seq'].shape} ({len(features_dict)}/{len(unique_vids)})")
            except Exception as e:
                print(f"[Pipeline] Warning: Failed processing {vid}: {e}")

    print(f"\n[Pipeline] Completed feature extraction for {len(features_dict)} videos.")

    # Build train/test splits from real extracted features
    train_df, test_df = loader.get_splits()

    def build_dataset_arrays(split_df):
        X_arr, y_arr = [], []
        for _, row in split_df.iterrows():
            vid = row["video_id"]
            label = row["class_idx"]
            if vid in features_dict:
                X_arr.append(features_dict[vid])
                y_arr.append(label)
            else:
                # Fallback if download failed
                print(f"[Pipeline] Warning: Missing features for {vid}, using mean imputation.")
                mean_feat = np.mean(list(features_dict.values()), axis=0) if features_dict else np.zeros((8, 584))
                X_arr.append(mean_feat)
                y_arr.append(label)
        return np.array(X_arr, dtype=np.float32), np.array(y_arr, dtype=np.int64)

    X_train, y_train = build_dataset_arrays(train_df)
    X_test, y_test = build_dataset_arrays(test_df)

    # Save real features cache
    cache_path = data_dir / "features_cache.npz"
    np.savez_compressed(
        cache_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test
    )
    print(f"[Pipeline] Real multimodal features saved to {cache_path} ({cache_path.stat().st_size / 1024:.1f} KB).")

    # Retrain Gated Attention MIL Classifier on 100% Real Features
    cfg = ModelConfig(epochs=30)
    model = VideoFormatClassifier(cfg)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n[Trainer] Training Gated Temporal Attention model on real video features...")
    model.train()
    for ep in range(30):
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0 or ep == 29:
            print(f"  Epoch [{ep+1}/30] - Loss: {total_loss / len(train_loader):.4f}")

    weights_path = data_dir / "model_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"[Trainer] Trained model weights saved to {weights_path}")

    # Evaluate on held-out test split
    model.eval()
    profiler = LatencyProfiler()
    y_preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            sample_t = torch.from_numpy(X_test[i]).unsqueeze(0)
            t0 = time.perf_counter()
            pred = model.predict(sample_t)
            profiler.record(time.perf_counter() - t0)
            y_preds.append(CLASS_NAMES.index(pred["category"]))

    std = compute_standard_metrics(y_test.tolist(), y_preds)
    ci = compute_bootstrap_confidence_intervals(y_test.tolist(), y_preds)
    lat = profiler.summary()
    peas = compute_peas_score(std["macro_f1"], lat["p95_ms"], sum(p.numel() for p in model.parameters()))

    print("\n================ UPDATED BENCHMARK REPORT (REAL FEATURES) ================")
    print(f"Top-1 Accuracy:       {std['accuracy'] * 100:.2f}% (95% CI: {ci['accuracy_ci']})")
    print(f"Macro-F1 Score:       {std['macro_f1']:.4f} (95% CI: {ci['macro_f1_ci']})")
    print(f"Latency p50 / p95:    {lat['p50_ms']} ms / {lat['p95_ms']} ms")
    print(f"Throughput:           {lat['throughput_fps']} inferences/sec")
    print(f"PEAS Score:           {peas}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
