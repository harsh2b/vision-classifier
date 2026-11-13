import cv2
import os
import json
import base64

videos = {
    'Sports_HighMotion': 'task_1_video_classifier/data/videos/913JhnW28U4.mp4',
    'Gameplay_Esports': 'task_1_video_classifier/data/videos/LlajSKnbcGk.mp4',
    'Podcast_TalkingHead': 'task_1_video_classifier/data/videos/c0siCya457M.mp4',
    'Wildlife_Nature': 'task_1_video_classifier/data/videos/jkNxmTrrZSk.mp4',
    'LifeRecord_Egocentric': 'task_1_video_classifier/data/videos/7iXM5aq53Ts.mp4',
    'Performance_Narrative': 'task_1_video_classifier/data/videos/YPaLGP_xr-w.mp4'
}
out_dir = 'c:/Desktop/Table/Job/assets/keyframes'
os.makedirs(out_dir, exist_ok=True)

results = {}
for cat, path in videos.items():
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps
    indices = [int(i * (total_frames - 1) / 7) for i in range(8)]
    cat_dir = os.path.join(out_dir, cat)
    os.makedirs(cat_dir, exist_ok=True)
    frames_info = []
    for idx_num, f_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (200, 112))
            f_name = f'frame_{idx_num}.jpg'
            f_path = os.path.join(cat_dir, f_name)
            cv2.imwrite(f_path, frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            _, buf = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
            frames_info.append({'idx': idx_num, 'time_sec': round(f_idx / fps, 1), 'b64': b64})
    cap.release()
    results[cat] = {'duration_sec': round(duration, 1), 'frames': frames_info}

with open('c:/Desktop/Table/Job/assets/keyframes/manifest.json', 'w') as f:
    json.dump(results, f)

print('Success: extracted keyframes for all categories.')
