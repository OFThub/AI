import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
from collections import OrderedDict
import os

###############################################################################
### Centroid Tracker Class
###############################################################################

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()      
        self.disappeared = OrderedDict()  
        self.info = {}                    
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, cls_name):
        oid = self.nextObjectID
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.info[oid] = {'class': cls_name, 'counted': False, 'history':[centroid]}
        self.nextObjectID += 1
        return oid

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        del self.info[oid]

    def update(self, rects, class_names):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects, self.info

        inputCentroids = np.array(rects)

        if len(self.objects) == 0:
            for i, c in enumerate(inputCentroids):
                self.register(tuple(c), class_names[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(np.array(objectCentroids)[:,None,:] - inputCentroids[None,:,:], axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                oid = objectIDs[row]
                self.objects[oid] = tuple(inputCentroids[col])
                self.disappeared[oid] = 0
                self.info[oid]['history'].append(tuple(inputCentroids[col]))
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])) - usedRows
            unusedCols = set(range(0, D.shape[1])) - usedCols

            for row in unusedRows:
                oid = objectIDs[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for col in unusedCols:
                self.register(tuple(inputCentroids[col]), class_names[col])

        return self.objects, self.info

DEF_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}  

###############################################################################
### Main Function
###############################################################################

def main(video_path=0,
         output_dir="./yolo_cars",
         yolo_weights="./yolo_cars/yolov8n.pt",
         conf_thresh=0.35):

    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(yolo_weights)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video açılamadı"

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_line = H // 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    output_video_path = os.path.join(output_dir, "out.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))


    tracker = CentroidTracker(max_disappeared=30, max_distance=80)

    total_red_path = 0
    total_green_path = 0
    total_blue_path = 0

    logfile = os.path.join(output_dir,f"vehicle_log_{int(time.time())}.csv")

    csvf = open(logfile, mode='w', newline='', encoding='utf-8')
    csvw = csv.writer(csvf)
    csvw.writerow(["timestamp", "frame", "object_id", "class", "direction"])

    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = model(frame, imgsz=640, conf=conf_thresh)[0]  
            boxes = []
            class_names = []
            crops = []

            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls.cpu().numpy().item())
                    conf = float(box.conf.cpu().numpy().item())
                    name = model.names.get(cls_id, str(cls_id))
                    if name not in DEF_VEHICLE_CLASSES:
                        continue
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    boxes.append((x1, y1, x2, y2, cx, cy))
                    class_names.append(name)
                    crop = frame[max(0,y1):y2, max(0,x1):x2]
                    crops.append(crop)

            rects = [(b[4], b[5]) for b in boxes] 
            objects, info = tracker.update(rects, class_names)

            for oid, centroid in objects.items():
                matched_idx = None
                for i, b in enumerate(boxes):
                    if abs(b[4] - centroid[0]) < 10 and abs(b[5] - centroid[1]) < 10:
                        matched_idx = i
                        break

                if matched_idx is not None:
                    x1,y1,x2,y2,cx,cy = boxes[matched_idx]
                    cls_name = class_names[matched_idx]
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame, f"ID {oid} {cls_name}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                else:
                    cx, cy = int(centroid[0]), int(centroid[1])
                    cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

###############################################################################
### Count Crossing
###############################################################################

                red_line_y    = center_line + 50  
                green_line_y  = center_line + 50  
                blue_line_y   = center_line + 50  
                
                red_line_x    = 0 
                green_line_x  = 250  
                blue_line_x   = 400 

                hist = info[oid]['history']
                direction = None

                if len(hist) >= 2 and not info[oid]['counted']:
                    prev_y = hist[-2][1]
                    cur_y  = hist[-1][1]
                    cur_x  = hist[-1][0]

                    if prev_y < red_line_y <= cur_y and red_line_x < cur_x < red_line_x + 250:
                        direction = "kirmizi_serit"
                        total_red_path += 1
                        info[oid]['counted'] = True

                    elif prev_y < green_line_y <= cur_y and green_line_x < cur_x < green_line_x + 150:
                        direction = "yesil_serit"
                        total_green_path += 1
                        info[oid]['counted'] = True

                    elif prev_y < blue_line_y <= cur_y and blue_line_x < cur_x < blue_line_x + 240:
                        direction = "mavi_serit"
                        total_blue_path += 1
                        info[oid]['counted'] = True


                if direction is not None:
                    timestamp = time.time() - start_time
                    csvw.writerow([f"{timestamp:.2f}", frame_idx, oid,
                                   info[oid]['class'], direction])
                    csvf.flush()

                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.putText(frame, f"ID{oid}", (cx-10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

###############################################################################
### Draw Lines and Display Counts
###############################################################################

            cv2.line(frame, (0, center_line+50), (250, center_line+50), (0,0,255), 2)
            cv2.line(frame, (250, center_line+50), (400, center_line+50), (0,255,0), 2)
            cv2.line(frame, (400, center_line+50), (640, center_line+50), (255,0,0), 2)
            cv2.putText(frame, f"Kirmizi Yoldan Gecenler : {total_red_path}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            cv2.putText(frame, f"Yesil Yoldan Gecenler : {total_green_path}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.putText(frame, f"Mavi Yoldan Gecenler : {total_blue_path}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2)
            cv2.putText(frame, f"Toplam: {total_red_path + total_green_path + total_blue_path}", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),2)

            cv2.imshow("Vehicle Counter", frame)
            if out:
                out.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        if out:
            out.release()
        csvf.close()
        cv2.destroyAllWindows()
        print("Log saved to:", logfile)
        print("Final counts - Kirmizi Yoldan Gecenler :", total_red_path,
              "Yesil Yoldan Gecenler :", total_green_path,
              "Mavi Yoldan Gecenler :", total_blue_path,
              "Toplam:", total_red_path + total_green_path + total_blue_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 Araç Tespiti")

    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Video dosyası yolu veya kamera indeksi (0, 1, ...)"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./yolo_cars",
        help="Çıktı klasörü (out.mp4 burada kaydedilir)"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="./yolo_cars/yolov8n.pt",
        help="YOLOv8 ağırlık dosyası yolu"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold"
    )

    args = parser.parse_args()

    # 🎥 Kamera mı dosya mı?
    video_source = int(args.video) if args.video.isdigit() else args.video

    main(
        video_path=video_source,
        output_dir=args.outdir,
        yolo_weights=args.weights,
        conf_thresh=args.conf
    )