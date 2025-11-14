# main.py
import cv2
from detector import PersonDetector
from tracker_wrapper import PeopleTracker
from config import YOLO_MODEL_PATH, DETECTION_CONFIDENCE, MIN_BOX_AREA, ROI_REL, BUCKETS


def point_in_roi(xc, yc, roi_abs):
    x1, y1, x2, y2 = roi_abs
    return x1 <= xc <= x2 and y1 <= yc <= y2


def choose_bucket(duration_sec: float):
    """
    Возвращает имя интервала (label) из BUCKETS или None.
    BUCKETS = [(start,end,label), ...]
    """
    for start, end, label in BUCKETS:
        if start <= duration_sec < end:
            return label
    return None


def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] FPS: {fps:.2f}, size: {width}x{height}")

    # считаем абсолютные координаты ROI
    x1_rel, y1_rel, x2_rel, y2_rel = ROI_REL
    roi_abs = (
        int(width * x1_rel),
        int(height * y1_rel),
        int(width * x2_rel),
        int(height * y2_rel),
    )
    print(f"[INFO] ROI (abs): {roi_abs}")

    detector = PersonDetector(
        model_path=YOLO_MODEL_PATH,
        conf_threshold=DETECTION_CONFIDENCE,
        min_box_area=MIN_BOX_AREA,
    )
    tracker = PeopleTracker()

    frame_idx = 0

    # Состояние по трекам внутри ROI:
    # track_state[track_id] = {
    #   "in_roi": bool,
    #   "enter_frame": int,
    #   "last_in_roi_frame": int,
    # }
    track_state = {}

    # список длительностей эпизодов
    episodes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # отметим кто в этом кадре был внутри ROI
        current_ids_in_roi = set()

        for tr in tracks:
            tid = tr["id"]
            l, t, r, b = tr["ltrb"]
            xc = (l + r) / 2
            yc = (t + b) / 2

            inside = point_in_roi(xc, yc, roi_abs)

            if tid not in track_state:
                track_state[tid] = {
                    "in_roi": False,
                    "enter_frame": None,
                    "last_in_roi_frame": None,
                }
            st = track_state[tid]

            if inside:
                current_ids_in_roi.add(tid)
                if not st["in_roi"]:
                    # только что вошёл
                    st["in_roi"] = True
                    st["enter_frame"] = frame_idx
                st["last_in_roi_frame"] = frame_idx
            else:
                # был внутри и вышел — завершаем эпизод
                if st["in_roi"]:
                    st["in_roi"] = False
                    enter_f = st["enter_frame"]
                    exit_f = st["last_in_roi_frame"]
                    if enter_f is not None and exit_f is not None:
                        dur_sec = (exit_f - enter_f + 1) / fps
                        episodes.append(dur_sec)
                    st["enter_frame"] = None
                    st["last_in_roi_frame"] = None

        # (опционально) можно рисовать ROI и треки для дебага:
        # x1, y1, x2, y2 = roi_abs
        # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        # for tr in tracks:
        #     l,t,r,b = map(int, tr["ltrb"])
        #     cv2.rectangle(frame, (l,t), (r,b), (255,0,0), 1)
        #     cv2.putText(frame, str(tr["id"]), (l,t-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        # cv2.imshow("debug", frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    cap.release()
    # cv2.destroyAllWindows()

    # Закрываем незавершённые эпизоды (если кто-то остался в ROI к концу видео)
    for tid, st in track_state.items():
        if st["in_roi"] and st["enter_frame"] is not None and st["last_in_roi_frame"] is not None:
            dur_sec = (st["last_in_roi_frame"] - st["enter_frame"] + 1) / fps
            episodes.append(dur_sec)

    # Теперь у нас есть список эпизодов (визитов в ROI)
    # Считаем статистику по интервалам
    bucket_counts = {label: 0 for _, _, label in BUCKETS}
    for dur in episodes:
        label = choose_bucket(dur)
        if label:
            bucket_counts[label] += 1

    stats = {
        "total_episodes": len(episodes),
        "bucket_counts": bucket_counts,
        "episodes_durations": episodes,
        "roi_abs": roi_abs,
    }
    return stats


if __name__ == "__main__":
    video_path = "videos/test1.mp4"

    stats = process_video(video_path)

    print("=== Статистика по 5-минутному видео ===")
    print(f"Всего эпизодов (визитов в зону): {stats['total_episodes']}")
    print("Распределение по интервалам:")
    for _, _, label in BUCKETS:
        print(f"  {label}: {stats['bucket_counts'][label]} эпизод(ов)")

    print("\nДлительности эпизодов (сек):")
    for i, dur in enumerate(stats["episodes_durations"], start=1):
        print(f"  Episode {i}: {dur:.1f} сек")
