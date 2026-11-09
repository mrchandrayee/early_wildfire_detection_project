import json,time
from pathlib import Path
import cv2
from .detector import FireSmokeDetector
from .alerts import AlertTracker
from .visualizer import draw_tech_box,draw_hud,COLOR_PALETTE

class WildfirePipeline:
    def __init__(self,args):
        self.args=args
        self.cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
        self.detector=FireSmokeDetector(args.weights,self.cfg["confidence"],
            self.cfg["iou"],args.device)
        self.alerts=AlertTracker(self.cfg["window_size"],self.cfg["snapshot_cooldown_sec"])
        self.events=[]

    def run(self):
        source=int(self.args.source) if str(self.args.source).isdigit() else self.args.source
        cap=cv2.VideoCapture(source)
        if not cap.isOpened():raise RuntimeError(f"Cannot open source: {self.args.source}")
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_in=cap.get(cv2.CAP_PROP_FPS); fps_in=fps_in if fps_in and fps_in>1 else 25.0
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out=Path(self.args.output); out.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps_in,(w,h))
        if not writer.isOpened():raise RuntimeError(f"Cannot create output: {out}")
        snap_dir=Path(self.cfg["snapshot_dir"])
        if self.args.save_snapshots:snap_dir.mkdir(parents=True,exist_ok=True)
        frame_idx=0; fps_smooth=0.0; started=time.time()
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                frame_idx+=1; tick=time.time()
                detections=self.detector.detect(frame)
                state=self.alerts.update(detections)
                for det in detections:
                    name=det["class_name"]; key=name.lower()
                    color=COLOR_PALETTE.get(key,COLOR_PALETTE["default"])
                    draw_tech_box(frame,det["box"],name,det["conf"],color,
                                  self.cfg["box_thickness"],self.cfg["corner_ratio"])
                if self.alerts.should_save_snapshot(state["status"]):
                    event={"frame":frame_idx,"time_sec":round(frame_idx/fps_in,3),
                           **state}
                    self.events.append(event)
                    if self.args.save_snapshots:
                        cv2.imwrite(str(snap_dir/f"wildfire_{frame_idx:07d}.jpg"),frame)
                    print(f'[ALERT] {state["status"]} at {event["time_sec"]:.2f}s')
                dt=max(time.time()-tick,1e-6); current=1/dt
                fps_smooth=current if fps_smooth==0 else .9*fps_smooth+.1*current
                frame=draw_hud(frame,state["status"],state["fire_count"],
                    state["smoke_count"],fps_smooth,frame_idx,total,
                    show_detail=self.cfg["show_detail"])
                writer.write(frame)
                if self.args.display:
                    cv2.imshow("Early Wildfire Detection",frame)
                    if cv2.waitKey(1)&0xFF==ord("q"):break
        finally:
            cap.release();writer.release()
            if self.args.display:cv2.destroyAllWindows()
        report={"source":str(self.args.source),"frames_processed":frame_idx,
            "duration_sec":round(frame_idx/fps_in,3),"runtime_sec":round(time.time()-started,2),
            "events":self.events}
        report_path=out.with_name(out.stem+"_events.json")
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"[DONE] Annotated video: {out}")
        print(f"[DONE] Event report: {report_path}")
