import cv2
from ultralytics import YOLO


def point_in_roi(xc, yc, roi):
    """
    roi = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = roi
    return x1 <= xc <= x2 and y1 <= yc <= y2


def process_video_with_roi(
    video_path: str,
    model_path: str = "yolov8n.pt",
    conf_thres: float = 0.6,
    min_box_area: float = 4000.0,
    roi=None,
):
    """
    Считает эпизоды присутствия людей в заданной зоне (ROI) и распределяет по интервалам.
    ROI задаём как (x1, y1, x2, y2) в координатах кадра.
    """

    # 1. FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"[INFO] FPS: {fps:.2f}, size: {width}x{height}")

    # Если roi не задан, берём центральную область кадра как пример
    if roi is None:
        roi = (
            int(width * 0.25),
            int(height * 0.25),
            int(width * 0.75),
            int(height * 0.75),
        )
    print(f"[INFO] ROI: {roi}")

    model = YOLO(model_path)

    results_gen = model.track(
        source=video_path,
        conf=conf_thres,
        classes=[0],
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    frame_idx = 0

    # для каждого track_id храним состояние:
    # in_roi: сейчас внутри ROI или нет
    # enter_frame: с какого кадра находится внутри
    # last_in_roi_frame: последний кадр, когда ещё был в ROI
    state = {}

    # список эпизодов (duration_sec)
    episodes = []

    for res in results_gen:
        frame_idx += 1
        boxes = res.boxes

        # отмечаем, кто в этом кадре был в ROI
        seen_in_this_frame = set()

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.tolist()

            for tid, box in zip(ids, xyxy):
                tid = int(tid)
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                area = w * h
                if area < min_box_area:
                    continue

                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2

                inside = point_in_roi(xc, yc, roi)

                # инициализируем состояние
                if tid not in state:
                    state[tid] = {
                        "in_roi": False,
                        "enter_frame": None,
                        "last_in_roi_frame": None,
                    }

                st = state[tid]

                if inside:
                    seen_in_this_frame.add(tid)
                    if not st["in_roi"]:
                        # только что вошёл
                        st["in_roi"] = True
                        st["enter_frame"] = frame_idx
                    st["last_in_roi_frame"] = frame_idx
                else:
                    # сейчас не внутри ROI — если он был внутри и вышел, завершаем эпизод
                    if st["in_roi"]:
                        st["in_roi"] = False
                        enter_f = st["enter_frame"]
                        exit_f = st["last_in_roi_frame"]
                        if enter_f is not None and exit_f is not None:
                            dur_sec = (exit_f - enter_f + 1) / fps
                            episodes.append(dur_sec)
                        st["enter_frame"] = None
                        st["last_in_roi_frame"] = None

        # тут можно добавить логику обработки треков, которые пропали совсем
        # (но для короткого 5-минутного видео можно не усложнять)

    # после прохода по видео — закрываем все незавершённые эпизоды
    for tid, st in state.items():
        if st["in_roi"] and st["enter_frame"] is not None and st["last_in_roi_frame"] is not None:
            dur_sec = (st["last_in_roi_frame"] - st["enter_frame"] + 1) / fps
            episodes.append(dur_sec)

    # теперь episodes = список "визитов" людей в зону ROI
    # распределяем по интервалам
    buckets = {
        "3-10": 0,
        "10-20": 0,
        "20-60": 0,
        "60-180": 0,
        "180+": 0,
    }
    fast_pass = 0
    for dur in episodes:
        if dur < 3:
            fast_pass += 1
        elif 3 <= dur < 10:
            buckets["3-10"] += 1
        elif 10 <= dur < 20:
            buckets["10-20"] += 1
        elif 20 <= dur < 60:
            buckets["20-60"] += 1
        elif 60 <= dur < 180:
            buckets["60-180"] += 1
        else:
            buckets["180+"] += 1

    stats = {
        "total_episodes": len(episodes),
        "fast_pass_under_3s": fast_pass,
        "dwell_buckets": buckets,
        "episodes_durations": episodes,
        "roi": roi,
    }
    return stats


if __name__ == "__main__":
    video_path = "videos/test1.mp4"

    stats = process_video_with_roi(video_path)

    print("=== Эпизоды в зоне ROI ===")
    print(f"Всего эпизодов: {stats['total_episodes']}")
    print(f"Быстрый проход (<3 сек): {stats['fast_pass_under_3s']}")
    print("Задержались на время:")
    for k, v in stats["dwell_buckets"].items():
        print(f"  {k} сек: {v} эпизод(ов)")

    print("\nДлительности эпизодов:")
    for i, dur in enumerate(stats["episodes_durations"], start=1):
        print(f"  Episode {i}: {dur:.1f} сек")
