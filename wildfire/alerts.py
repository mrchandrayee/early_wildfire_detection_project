from collections import deque
import time

class AlertTracker:
    """Sliding-window fire/smoke smoothing and snapshot cooldown."""
    def __init__(self,window_size=10,alert_cooldown=3.0):
        self.window=deque(maxlen=int(window_size))
        self.last_alert_time=0.0
        self.alert_cooldown=float(alert_cooldown)

    def update(self,detections):
        fire_count=sum(1 for d in detections if d["class_name"].lower()=="fire")
        smoke_count=sum(1 for d in detections if d["class_name"].lower()=="smoke")
        self.window.append((fire_count,smoke_count))
        avg_fire=sum(x[0] for x in self.window)/len(self.window)
        avg_smoke=sum(x[1] for x in self.window)/len(self.window)
        if avg_fire>=.5 and avg_smoke>=.5:
            status="CRITICAL"
        elif avg_fire>=.5:
            status="DANGER"
        elif avg_smoke>=.5:
            status="WARNING"
        else:
            status="SAFE"
        return {"status":status,"fire_count":fire_count,"smoke_count":smoke_count,
                "avg_fire":avg_fire,"avg_smoke":avg_smoke}

    def should_save_snapshot(self,status):
        now=time.time()
        if status in ("WARNING","DANGER","CRITICAL") and now-self.last_alert_time>=self.alert_cooldown:
            self.last_alert_time=now
            return True
        return False
