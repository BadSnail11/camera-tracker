# step2_merge_tracks.py
import pickle
import numpy as np
from itertools import combinations
from math import sqrt


TRACKS_FILE = "tracks.pkl"

# максимально допустимый разрыв между треками (секунд)
MAX_GAP_SEC = 3.0

# минимальное сходство по фичам (cosine similarity) чтобы считать «тем же человеком»
MIN_COS_SIM = 0.8

# максимальное расстояние между центрами боксов,
# чтобы два трека могли быть одним человеком (в пикселях)
MAX_CENTER_DIST = 150.0


def cosine_sim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a.dot(b) / (na * nb))


def center_of_bbox(bbox):
    l, t, r, b = bbox
    return ( (l + r) / 2.0, (t + b) / 2.0 )


def dist(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def main():
    with open(TRACKS_FILE, "rb") as f:
        data = pickle.load(f)

    fps = data["fps"]
    raw_tracks = data["tracks"]

    print(f"[INFO] FPS: {fps:.2f}, треков: {len(raw_tracks)}")

    max_gap_frames = int(MAX_GAP_SEC * fps)

    # 1. Предобработаем треки
    # создадим список объектов для удобства
    tracks = []
    for tid, tr in raw_tracks.items():
        frames = tr["frames"]
        bboxes = tr["bboxes"]
        feats = tr["features"]

        if len(frames) == 0:
            continue

        start = min(frames)
        end = max(frames)

        # сортируем по frame_idx
        order = np.argsort(frames)
        frames = [frames[i] for i in order]
        bboxes = [bboxes[i] for i in order]
        feats = [feats[i] for i in order]

        # средняя фича
        mean_feat = np.mean(np.array(feats), axis=0)

        tracks.append({
            "tid": tid,
            "start": start,
            "end": end,
            "frames": frames,
            "bboxes": bboxes,
            "mean_feat": mean_feat,
        })

    print(f"[INFO] После фильтрации осталось треков: {len(tracks)}")

    # 2. Готовим граф похожести треков:
    # будем хранить список "можно склеить" (i,j)
    n = len(tracks)
    merge_edges = []

    for i, j in combinations(range(n), 2):
        ti = tracks[i]
        tj = tracks[j]

        # 2.1. Временной критерий: треки не должны пересекаться и gap не слишком большой
        if ti["end"] >= tj["start"] and tj["end"] >= ti["start"]:
            # перекрываются по времени → это почти наверняка разные люди
            continue

        # кто раньше, кто позже
        if ti["end"] < tj["start"]:
            gap = tj["start"] - ti["end"]
            first = ti
            second = tj
        else:
            gap = ti["start"] - tj["end"]
            first = tj
            second = ti

        if gap > max_gap_frames:
            continue

        # 2.2. Пространственный критерий: центры концов и начала треков близки
        center1 = center_of_bbox(first["bboxes"][-1])
        center2 = center_of_bbox(second["bboxes"][0])
        if dist(center1, center2) > MAX_CENTER_DIST:
            continue

        # 2.3. Критерий внешнего вида (re-id)
        sim = cosine_sim(first["mean_feat"], second["mean_feat"])
        if sim < MIN_COS_SIM:
            continue

        merge_edges.append((i, j, sim))

    print(f"[INFO] Кандидатов на склейку: {len(merge_edges)}")

    # 3. Жадно склеиваем по убыванию similarity
    # будем хранить "лидеров" кластеров
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # сортируем ребра по similarity убыванию
    merge_edges.sort(key=lambda e: e[2], reverse=True)
    for i, j, sim in merge_edges:
        union(i, j)

    # 4. Строим кластеры
    clusters = {}
    for idx in range(n):
        r = find(idx)
        clusters.setdefault(r, []).append(idx)

    print(f"[INFO] Кластеров (людей): {len(clusters)}")

    # 5. Для каждого кластера считаем «человека»
    people_durations = []
    for cid, idxs in clusters.items():
        person_tracks = [tracks[i] for i in idxs]
        start = min(t["start"] for t in person_tracks)
        end = max(t["end"] for t in person_tracks)
        dur_sec = (end - start + 1) / fps
        people_durations.append(dur_sec)

    people_durations.sort()
    print("=== Итог по людям ===")
    print(f"Всего людей (кластеров): {len(people_durations)}")
    for i, d in enumerate(people_durations, 1):
        print(f"  Person {i}: {d:.1f} сек")

    # 6. Разложим по интервалам, как ты хотел
    buckets = {
        "3-10": 0,
        "10-20": 0,
        "20-60": 0,
        "60-180": 0,
        "180+": 0,
    }
    fast_pass = 0

    for d in people_durations:
        if d < 3:
            fast_pass += 1
        elif 3 <= d < 10:
            buckets["3-10"] += 1
        elif 10 <= d < 20:
            buckets["10-20"] += 1
        elif 20 <= d < 60:
            buckets["20-60"] += 1
        elif 60 <= d < 180:
            buckets["60-180"] += 1
        else:
            buckets["180+"] += 1

    print("\n=== Статистика по интервалам ===")
    print(f"Быстро прошли (<3 c): {fast_pass}")
    for label, val in buckets.items():
        print(f"  {label}: {val} чел")


if __name__ == "__main__":
    main()
