# tracker_wrapper.py
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict


class PeopleTracker:
    """
    Обёртка над DeepSort:
    На вход — детекции YOLO, на выход — треки:
        [{"id": int, "ltrb": [l,t,r,b]}]
    """
    def __init__(self):
        # параметры можно тонко тюнить под твоё видео
        self.tracker = DeepSort(
            max_age=30,           # сколько кадров можно "не видеть" трек
            n_init=3,             # сколько раз надо подряд увидеть, чтобы подтвердить трек
            max_iou_distance=0.7, # порог IoU для ассоциации
        )

    def update(self, detections, frame) -> List[Dict]:
        """
        detections: список ([l,t,w,h], conf, "person")
        возвращает треки: [{"id": track_id, "ltrb": [l,t,r,b]}]
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        output_tracks = []

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            track_id = tr.track_id
            l, t, r, b = tr.to_ltrb()  # left, top, right, bottom
            output_tracks.append({
                "id": int(track_id),
                "ltrb": [float(l), float(t), float(r), float(b)],
            })
        return output_tracks
