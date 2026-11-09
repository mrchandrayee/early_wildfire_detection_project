from pathlib import Path
import torch
from ultralytics import YOLO

DEFAULT_CLASS_NAMES={0:"Fire",1:"Smoke"}

class FireSmokeDetector:
    def __init__(self,weights_path="fireetect-11s.pt",conf=.25,iou=.45,device="auto"):
        if not Path(weights_path).exists() and ("/" in str(weights_path) or "\\" in str(weights_path)):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        if device=="auto":
            device="cuda" if torch.cuda.is_available() else "cpu"
        elif device=="cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA unavailable; using CPU")
            device="cpu"
        self.device=device
        self.conf=float(conf); self.iou=float(iou)
        print(f"[INFO] Loading Fire & Smoke model: {weights_path} on {device}")
        self.model=YOLO(weights_path)
        self.names=self.model.names
        print(f"[INFO] Model loaded. Classes: {self.names}")

    def detect(self,frame):
        results=self.model.predict(frame,conf=self.conf,iou=self.iou,
                                   device=self.device,verbose=False)
        detections=[]
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())
                cls_id=int(box.cls[0].cpu().item())
                conf_score=float(box.conf[0].cpu().item())
                if isinstance(self.names,dict):
                    cls_name=self.names.get(cls_id,DEFAULT_CLASS_NAMES.get(cls_id,f"Class {cls_id}"))
                else:
                    cls_name=self.names[cls_id] if cls_id<len(self.names) else f"Class {cls_id}"
                detections.append({"box":(x1,y1,x2,y2),"class_id":cls_id,
                                   "class_name":str(cls_name),"conf":conf_score})
        return detections
