import cv2

COLOR_PALETTE={"fire":(0,80,255),"smoke":(0,220,255),"default":(0,255,255)}
STATUS_COLORS={"SAFE":(50,180,50),"WARNING":(0,165,255),"DANGER":(0,0,230),"CRITICAL":(0,0,150)}

def draw_tech_box(img,box,label_name,conf,color=None,thickness=4,corner_ratio=.15):
    if color is None: color=COLOR_PALETTE.get(label_name.lower(),COLOR_PALETTE["default"])
    x1,y1,x2,y2=map(int,box); h,w=img.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
    if x2<=x1 or y2<=y1:return
    scale=max(1.0,min(w,h)/720.0)
    thick=max(3,int(thickness*scale)); glow=thick*2
    corner_len=max(14,int(min(x2-x1,y2-y1)*corner_ratio))
    # White glow outline, then colored outline and L-shaped corner accents.
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),thick+1,cv2.LINE_AA)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,thick,cv2.LINE_AA)
    corners=[((x1,y1),(x1+corner_len,y1)),((x1,y1),(x1,y1+corner_len)),
             ((x2,y1),(x2-corner_len,y1)),((x2,y1),(x2,y1+corner_len)),
             ((x1,y2),(x1+corner_len,y2)),((x1,y2),(x1,y2-corner_len)),
             ((x2,y2),(x2-corner_len,y2)),((x2,y2),(x2,y2-corner_len))]
    for p1,p2 in corners:
        cv2.line(img,p1,p2,(255,255,255),glow,cv2.LINE_AA)
    for p1,p2 in corners:
        cv2.line(img,p1,p2,color,thick,cv2.LINE_AA)
    font=cv2.FONT_HERSHEY_SIMPLEX; fs=.75*scale; ft=max(2,int(2.5*scale))
    text=f"{label_name.upper()} ({conf:.0%})"
    (tw,th),base=cv2.getTextSize(text,font,fs,ft)
    pad_x=max(8,int(10*scale)); pad_y=max(5,int(7*scale))
    badge_y=max(th+pad_y*2,y1)
    badge_x=x1; bx2=min(w-1,badge_x+tw+pad_x*2)
    by1=max(0,badge_y-th-pad_y*2); by2=min(h-1,badge_y)
    cv2.rectangle(img,(badge_x,by1),(bx2,by2),(0,0,0),-1)
    cv2.rectangle(img,(badge_x,by1),(bx2,by2),color,1)
    luminance=.299*color[2]+.587*color[1]+.114*color[0]
    txt_color=(0,0,0) if luminance>150 else (255,255,255)
    cv2.putText(img,text,(badge_x+pad_x,by2-pad_y),font,fs,txt_color,ft,cv2.LINE_AA)

def draw_hud(img,status,fire_count,smoke_count,fps,frame_idx,total_frames,
             top_offset=0,show_detail=True,blink_hz=0.0):
    h,w=img.shape[:2]; scale=max(1.0,min(h/720.0,1.4))
    color=STATUS_COLORS.get(status,(150,50,50))
    banner_h=int(45*scale); y1=max(0,int(top_offset))
    y2=min(h-1,y1+banner_h)
    overlay=img.copy()
    cv2.rectangle(overlay,(0,y1),(w-1,y2),color,-1)
    cv2.addWeighted(overlay,.75,img,.25,0,img)
    if status=="SAFE": banner="SYSTEM NORMAL - NO FIRE / SMOKE DETECTED"
    elif status=="WARNING": banner=f"WARNING: SMOKE DETECTED ({smoke_count})"
    elif status=="DANGER": banner=f"DANGER: FIRE DETECTED ({fire_count})"
    else: banner=f"CRITICAL: FIRE + SMOKE DETECTED ({fire_count}/{smoke_count})"
    font=cv2.FONT_HERSHEY_SIMPLEX; fs=.7*scale; ft=max(2,int(2*scale))
    (tw,th),_=cv2.getTextSize(banner,font,fs,ft)
    cv2.putText(img,banner,(max(8,(w-tw)//2),y1+(banner_h+th)//2),font,fs,(255,255,255),ft,cv2.LINE_AA)
    if show_detail:
        cw,ch=int(250*scale),int(130*scale)
        x1,y1=int(10*scale),min(h-1,int(top_offset)+banner_h+int(10*scale))
        x2,y2=min(w-1,x1+cw),min(h-1,y1+ch)
        card=img.copy(); cv2.rectangle(card,(x1,y1),(x2,y2),(20,20,20),-1)
        cv2.addWeighted(card,.65,img,.35,0,img)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        lines=["WILDFIRE DETECTION",f"STATUS: {status}",
               f"FIRE: {fire_count}",f"SMOKE: {smoke_count}",
               f"FPS: {fps:.1f} | FRAME: {frame_idx}/{total_frames or '?'}"]
        yy=y1+int(22*scale)
        for line in lines:
            cv2.putText(img,line,(x1+int(8*scale),yy),font,.48*scale,
                        (255,255,255),1,cv2.LINE_AA)
            yy+=int(21*scale)
    return img
