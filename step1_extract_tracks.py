# step1_extract_tracks.py
import cv2
import pickle
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


VIDEO_PATH = "videos/test1.mp4"
YOLO_MODEL = "yolov8m.pt"   # можно n/m/l - m уже поадекватнее
CONF_THRES = 0.6
MIN_BOX_AREA = 3000.0


def main():
    # 1. детектор
    yolo = YOLO(YOLO_MODEL)

    # 2. трекер с re-id
    tracker = DeepSort(
        max_age=50,
        n_init=3,
        max_iou_distance=0.7,
        embedder="mobilenet",
        half=True,
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] FPS: {fps:.2f}")

    frame_idx = 0

    # Для каждого track_id сохраняем:
    # tracks[tid] = {
    #   "frames": [frame_idx...],
    #   "bboxes": [[l,t,r,b]...],
    #   "features": [[f1,f2,...], ...]
    # }
    tracks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO детекция
        res = yolo(frame, conf=CONF_THRES, verbose=False)[0]

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                if int(cls) != 0:
                    continue  # не person
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                area = w * h
                if area < MIN_BOX_AREA:
                    continue
                detections.append(([x1, y1, w, h], float(conf), "person"))

        # DeepSort update
        ds_tracks = tracker.update_tracks(detections, frame=frame)

        for tr in ds_tracks:
            if not tr.is_confirmed():
                continue
            tid = int(tr.track_id)
            l, t, r, b = tr.to_ltrb()
            feat = tr.get_feature()  # эмбеддинг re-id

            if tid not in tracks:
                tracks[tid] = {
                    "frames": [],
                    "bboxes": [],
                    "features": [],
                }

            tracks[tid]["frames"].append(frame_idx)
            tracks[tid]["bboxes"].append([float(l), float(t), float(r), float(b)])
            tracks[tid]["features"].append(feat.tolist())

    cap.release()

    data = {
        "fps": fps,
        "tracks": tracks,
    }

    with open("tracks.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Сохранили {len(tracks)} треков в tracks.pkl")


if __name__ == "__main__":
    main()
