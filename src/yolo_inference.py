# minimal comments
from ultralytics import YOLO
import numpy as np

_model = None

def load_model(model_path: str, device: str = "cpu"):
    global _model
    if _model is None:
        _model = YOLO(model_path)
        try:
            _model.to(device)
        except Exception:
            pass
    return _model

def run_inference(model, image_np: np.ndarray, conf_thres: float = 0.25, imgsz: int = 640):
    # returns list of dicts: {"class_id", "class_name", "conf", "bbox":[x1,y1,x2,y2]}
    results = model.predict(source=image_np, imgsz=imgsz, conf=conf_thres, verbose=False)
    out = []
    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        # boxes.xyxy, boxes.cls, boxes.conf
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
        for i in range(len(xyxy)):
            bbox = xyxy[i].tolist()
            cid = int(cls_ids[i])
            conf = float(confs[i])
            out.append({"class_id": cid, "conf": conf, "bbox": bbox})
    return out
