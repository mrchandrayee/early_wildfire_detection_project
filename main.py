#!/usr/bin/env python3
"""Early wildfire / fire and smoke detection."""
import argparse
from wildfire.pipeline import WildfirePipeline

def parse_args():
    p=argparse.ArgumentParser(description="Early wildfire detection using YOLO fire/smoke model")
    p.add_argument("--source",default="0",help="Webcam index or video path")
    p.add_argument("--weights",default="fireetect-11s.pt",help="Custom YOLO fire/smoke checkpoint")
    p.add_argument("--output",default="output/wildfire_detection.mp4")
    p.add_argument("--config",default="config/settings.json")
    p.add_argument("--device",default="auto",help="auto, cpu, cuda, or CUDA index")
    p.add_argument("--display",action="store_true")
    p.add_argument("--save-snapshots",action="store_true")
    return p.parse_args()

if __name__=="__main__":
    WildfirePipeline(parse_args()).run()
