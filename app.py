from flask import Flask, render_template, Response, request, jsonify
import cv2, time, threading, os, tempfile, re, json
import pytesseract
from ultralytics import YOLO

app=Flask(__name__)
model=YOLO("yolo11n.pt")
cap=None
source_lock=threading.Lock()
registry_lock=threading.Lock()
events=[]
offline_mode=False
zone=(0.62,0.15,0.98,0.88)
TESSERACT_PATH=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd=TESSERACT_PATH
anpr_seen={}
track_history={}
REGISTRY_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"entity_registry.json")

VEHICLES={"car","motorcycle","bus","truck","bicycle"}
ANIMALS={"bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"}
DEFAULT_REGISTRY=[
 {"entity_id":"ENT-001","name":"Demo Person 01","category":"WANTED","type":"PERSON","identifier":"DEMO-PERSON-001","plate":"","status":"ACTIVE","description":"Synthetic SIH demonstration record"},
 {"entity_id":"ENT-002","name":"Demo Vehicle 01","category":"VEHICLE_WATCHLIST","type":"VEHICLE","identifier":"","plate":"DEMO1234","status":"ACTIVE","description":"Synthetic SIH demonstration vehicle"},
 {"entity_id":"ENT-003","name":"Authorized Personnel 01","category":"AUTHORIZED","type":"PERSON","identifier":"DEMO-AUTH-001","plate":"","status":"ACTIVE","description":"Synthetic authorized personnel record"}
]

def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE,"w",encoding="utf-8") as f: json.dump(DEFAULT_REGISTRY,f,indent=2)
    try:
        with open(REGISTRY_FILE,"r",encoding="utf-8") as f: data=json.load(f)
        return data if isinstance(data,list) else []
    except Exception:
        return []

def save_registry(data):
    with open(REGISTRY_FILE,"w",encoding="utf-8") as f: json.dump(data,f,indent=2)

registry=load_registry()

def add_event(kind,severity,detail):
    events.insert(0,{"time":time.strftime("%H:%M:%S"),"kind":kind,"severity":severity,"detail":detail})
    del events[30:]

def point_in_zone(x,y,w,h):
    x1,y1,x2,y2=zone
    return x1*w<=x<=x2*w and y1*h<=y<=y2*h

def normalize_plate(value):
    return re.sub(r"[^A-Z0-9]","",str(value or "").upper())

def registry_match(plate="",identifier=""):
    p=normalize_plate(plate); ident=str(identifier or "").strip().upper()
    with registry_lock:
        for entity in registry:
            if str(entity.get("status","ACTIVE")).upper()!="ACTIVE": continue
            ep=normalize_plate(entity.get("plate","")); ei=str(entity.get("identifier","")).strip().upper()
            if (p and ep and p==ep) or (ident and ei and ident==ei): return entity
    return None

def clean_plate_text(text): return re.sub(r"[^A-Z0-9]","",text.upper())

def read_plate(vehicle_crop):
    if vehicle_crop is None or vehicle_crop.size==0: return None,0
    gray=cv2.cvtColor(vehicle_crop,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    gray=cv2.bilateralFilter(gray,9,75,75)
    variants=[cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)]
    candidates=[]; gh,gw=gray.shape[:2]
    for binary in variants:
        contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h=cv2.boundingRect(c); ratio=w/float(h) if h else 0
            if 2.0<=ratio<=6.5 and w*h>=max(120,gw*gh*0.002):
                candidates.append(gray[max(0,y-2):min(gh,y+h+2),max(0,x-2):min(gw,x+w+2)])
    candidates.append(gray); best=None; best_score=0
    for candidate in candidates[:25]:
        if candidate.size==0: continue
        candidate=cv2.resize(candidate,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC)
        candidate=cv2.GaussianBlur(candidate,(3,3),0)
        _,candidate=cv2.threshold(candidate,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        try: raw=pytesseract.image_to_string(candidate,config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        except Exception: return None,0
        plate=clean_plate_text(raw)
        if 5<=len(plate)<=12:
            score=min(100,len(plate)*9)
            if any(ch.isdigit() for ch in plate): score+=15
            if any(ch.isalpha() for ch in plate): score+=15
            if score>best_score: best_score,best=score,plate
    return best,best_score

def frames():
    global cap
    if cap is None: cap=cv2.VideoCapture(0)
    while True:
        ok,frame=cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,frame=cap.read()
            if not ok: time.sleep(.1); continue
        h,w=frame.shape[:2]; results=model.track(frame,conf=.45,verbose=False,device="cpu",persist=True,tracker="bytetrack.yaml")
        person_intrusion=False; detections=0
        for r in results:
            for b in r.boxes:
                cls=int(b.cls[0]); name=model.names[cls]; conf=float(b.conf[0]); track_id=int(b.id[0]) if b.id is not None else None
                x1,y1,x2,y2=map(int,b.xyxy[0]); detections+=1
                if track_id is not None:
                    hist=track_history.setdefault(track_id,[])
                    hist.append({"time":time.strftime("%H:%M:%S"),"class":name,"camera":"CAM-01","x":(x1+x2)//2,"y":(y1+y2)//2})
                    if len(hist)>50: del hist[:-50]
                if name=="person":
                    cx=(x1+x2)//2; cy=(y1+y2)//2; inside=point_in_zone(cx,cy,w,h)
                    label=f"PERSON #{track_id} {conf*100:.0f}%" if track_id is not None else f"PERSON {conf*100:.0f}%"; color=(0,0,255) if inside else (255,190,0)
                    if inside: person_intrusion=True
                elif name in VEHICLES:
                    label=f"{name.upper()} #{track_id} {conf*100:.0f}%" if track_id is not None else f"{name.upper()} {conf*100:.0f}%"; color=(0,220,100)
                    vehicle_crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]; plate,plate_conf=read_plate(vehicle_crop)
                    if plate and plate_conf>=45:
                        label=f"{name.upper()} | PLATE {plate} {plate_conf}%"; cv2.putText(frame,f"ANPR: {plate}",(x1,max(42,y1-28)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                        now=time.strftime("%H:%M:%S")
                        if anpr_seen.get(plate)!=now:
                            add_event("ANPR Detection","MEDIUM",f"{plate} detected on {name}")
                            anpr_seen[plate]=now
                        match=registry_match(plate=plate)
                        if match:
                            sev="HIGH" if match.get("category") in {"WANTED","VEHICLE_WATCHLIST","SECURITY_WATCHLIST"} else "LOW"
                            if anpr_seen.get("MATCH:"+plate)!=now:
                                add_event("Registry Match",sev,f"Potential registry match: {match.get('name')} ({match.get('category')})")
                                anpr_seen["MATCH:"+plate]=now
                elif name in ANIMALS:
                    label=f"{name.upper()} {conf*100:.0f}%"; color=(255,170,0)
                else: continue
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2); cv2.putText(frame,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        zx1,zy1,zx2,zy2=int(zone[0]*w),int(zone[1]*h),int(zone[2]*w),int(zone[3]*h)
        cv2.rectangle(frame,(zx1,zy1),(zx2,zy2),(0,0,255),2); cv2.putText(frame,"RESTRICTED ZONE",(zx1,zy1-8),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
        if person_intrusion:
            cv2.rectangle(frame,(0,0),(w-1,h-1),(0,0,255),5); cv2.putText(frame,"INTRUSION ALERT",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            if not events or events[0]["kind"]!="Restricted Zone Intrusion" or events[0]["time"]!=time.strftime("%H:%M:%S"): add_event("Restricted Zone Intrusion","HIGH","Person detected inside virtual fence")
        cv2.putText(frame,f"IBVAP | AI detections: {detections}",(15,h-18),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,235,250),2)
        ok,buf=cv2.imencode(".jpg",frame)
        if ok: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"

@app.get("/entities")
def get_entities():
    with registry_lock: return jsonify({"entities":registry})

@app.post("/entities")
def add_entity():
    global registry
    data=request.get_json(silent=True) or {}
    if not str(data.get("name","")).strip() or not str(data.get("category","")).strip() or not str(data.get("type","")).strip(): return jsonify({"ok":False,"error":"name, category and type are required"}),400
    with registry_lock:
        entity={"entity_id":str(data.get("entity_id") or f"ENT-{len(registry)+1:03d}"),"name":str(data.get("name","")).strip(),"category":str(data.get("category","UNKNOWN")).strip().upper(),"type":str(data.get("type","PERSON")).strip().upper(),"identifier":str(data.get("identifier","")).strip(),"plate":normalize_plate(data.get("plate","")),"status":str(data.get("status","ACTIVE")).strip().upper(),"description":str(data.get("description","")).strip()}
        if any(e.get("entity_id")==entity["entity_id"] for e in registry): return jsonify({"ok":False,"error":"Entity ID already exists"}),409
        registry.append(entity); save_registry(registry)
    add_event("Registry Updated","LOW",f"Added {entity['name']} ({entity['category']})"); return jsonify({"ok":True,"entity":entity})

@app.delete("/entities/<entity_id>")
def delete_entity(entity_id):
    global registry
    with registry_lock:
        old_len=len(registry); registry=[e for e in registry if e.get("entity_id")!=entity_id]
        if len(registry)==old_len: return jsonify({"ok":False,"error":"Entity not found"}),404
        save_registry(registry)
    add_event("Registry Updated","LOW",f"Removed {entity_id}"); return jsonify({"ok":True})

@app.get("/tracking")
def tracking():
    return jsonify({"entities":[{"track_id":tid,"class":hist[-1]["class"],"observations":hist[-20:]} for tid,hist in track_history.items() if hist]})

@app.get("/anpr-status")
def anpr_status(): return jsonify({"tesseract":os.path.exists(TESSERACT_PATH),"path":TESSERACT_PATH})

@app.route("/")
def index(): return render_template("index.html")
@app.route("/add-person")
def add_person_page(): return render_template("add_person.html")
@app.route("/add-vehicle")
def add_vehicle_page(): return render_template("add_vehicle.html")
@app.route("/video")
def video(): return Response(frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/upload")
def upload():
    global cap
    f=request.files.get("video")
    if not f: return jsonify({"ok":False,"error":"No video selected"}),400
    path=os.path.join(tempfile.gettempdir(),"ibvap_video.mp4"); f.save(path)
    with source_lock:
        if cap: cap.release()
        cap=cv2.VideoCapture(path)
    add_event("Video Source Changed","LOW","Uploaded video is now the active CCTV simulation"); return jsonify({"ok":True})

@app.get("/events")
def get_events(): return jsonify({"events":events,"offline":offline_mode})
@app.post("/offline")
def offline():
    global offline_mode
    offline_mode=not offline_mode; add_event("Network State","MEDIUM" if offline_mode else "LOW","Offline edge mode active" if offline_mode else "Network restored; synchronization available"); return jsonify({"offline":offline_mode})
@app.post("/clear")
def clear(): events.clear(); return jsonify({"ok":True})

import os

if _name=="main_":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
