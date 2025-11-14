# detector.py
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple


class PersonDetector:
    """
    Обёртка над YOLOv8: по кадру возвращает детекции людей в формате,
    который ожидает DeepSort:
        ([left, top, width, height], confidence, class_name)
    """
    def __init__(self, model_path: str, conf_threshold: float, min_box_area: float):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_box_area = min_box_area

    def detect(self, frame) -> List[Tuple[list, float, str]]:
        """
        Возвращает список детекций: ([l, t, w, h], conf, "person")
        """
        h, w, _ = frame.shape

        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(xyxy, confs, classes):
            # COCO класс 0 — person
            if cls != 0:
                continue

            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            if area < self.min_box_area:
                continue

            detections.append(([float(x1), float(y1), float(bw), float(bh)], float(conf), "person"))

        return detections
