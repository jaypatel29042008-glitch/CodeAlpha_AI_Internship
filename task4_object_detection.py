"""
TASK 4: Object Detection and Tracking
CodeAlpha AI Internship
Real-time detection with YOLOv8 + SORT-style centroid tracking via OpenCV
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
VIDEO_SOURCE  = 0           # 0 = webcam; replace with "path/to/video.mp4" for file
CONF_THRESH   = 0.45
IOU_THRESH    = 0.45
MAX_TRAIL_LEN = 30          # frames to keep tracking trail
MODEL_SIZE    = "yolov8n"   # nano – fastest; swap for yolov8s/m for more accuracy

# ── Lightweight centroid tracker ───────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared: int = 20):
        self.next_id       = 0
        self.objects       = {}           # id -> centroid
        self.disappeared   = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.trails        = defaultdict(lambda: deque(maxlen=MAX_TRAIL_LEN))

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bboxes):
        if not bboxes:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
                    if oid in self.trails:
                        del self.trails[oid]
            return {}

        new_centroids = [self._centroid(b) for b in bboxes]

        if not self.objects:
            for c in new_centroids:
                self.objects[self.next_id] = c
                self.next_id += 1
            result = {oid: c for oid, c in self.objects.items()}
            for oid, c in result.items():
                self.trails[oid].appendleft(c)
            return result

        obj_ids  = list(self.objects.keys())
        obj_cens = list(self.objects.values())

        D = np.linalg.norm(
            np.array(obj_cens)[:, None] - np.array(new_centroids)[None, :], axis=2
        )
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = obj_ids[r]
            self.objects[oid] = new_centroids[c]
            self.disappeared[oid] = 0
            self.trails[oid].appendleft(new_centroids[c])
            used_rows.add(r); used_cols.add(c)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for r in unused_rows:
            oid = obj_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                del self.objects[oid]; del self.disappeared[oid]
                if oid in self.trails: del self.trails[oid]

        for c in unused_cols:
            self.objects[self.next_id] = new_centroids[c]
            self.trails[self.next_id].appendleft(new_centroids[c])
            self.next_id += 1

        return {oid: c for oid, c in self.objects.items()}

# ── Colour palette per track ID ────────────────────────────────────────────────
_PALETTE = [
    (255, 99, 71), (135, 206, 235), (144, 238, 144), (255, 215, 0),
    (255, 182, 193), (0, 255, 127), (173, 216, 230), (255, 165, 0),
    (138, 43, 226), (0, 206, 209),
]

def id_color(oid: int):
    return _PALETTE[oid % len(_PALETTE)]

# ── Main loop ──────────────────────────────────────────────────────────────────
def run():
    model   = YOLO(f"{MODEL_SIZE}.pt")
    tracker = CentroidTracker(max_disappeared=25)
    cap     = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {VIDEO_SOURCE}")
        return

    print("[INFO] Press 'q' to quit, 's' to save screenshot.")
    frame_idx  = 0
    total_seen = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        # ── Detection ──────────────────────────────────────────────────────────
        results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]

        bboxes = []
        labels = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            bboxes.append((x1, y1, x2, y2))
            labels.append((label, conf))

        # ── Tracking ───────────────────────────────────────────────────────────
        tracked = tracker.update(bboxes)
        total_seen.update(tracked.keys())

        # ── Draw ───────────────────────────────────────────────────────────────
        for i, ((x1, y1, x2, y2), (label, conf)) in enumerate(zip(bboxes, labels)):
            # Find matching track id by closest centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if tracked:
                oid = min(tracked.keys(), key=lambda k: (tracked[k][0]-cx)**2 + (tracked[k][1]-cy)**2)
            else:
                oid = i
            color = id_color(oid)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label pill
            text = f"ID:{oid} {label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            # Trail
            trail = list(tracker.trails.get(oid, []))
            for j in range(1, len(trail)):
                alpha = 1 - j / len(trail)
                tc = tuple(int(c * alpha) for c in color)
                cv2.line(frame, trail[j-1], trail[j], tc, 2)

        # ── HUD ────────────────────────────────────────────────────────────────
        hud = (f"Frame: {frame_idx}  |  Active: {len(tracked)}  "
               f"|  Total tracked: {len(total_seen)}")
        cv2.rectangle(frame, (0, 0), (len(hud)*8 + 10, 28), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1)

        cv2.imshow("Object Detection & Tracking — CodeAlpha", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"screenshot_{frame_idx}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[SAVED] {fname}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Processed {frame_idx} frames. Unique objects tracked: {len(total_seen)}")


if __name__ == "__main__":
    run()
