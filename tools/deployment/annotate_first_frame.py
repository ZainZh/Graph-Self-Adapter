#!/usr/bin/env python3
"""Manual boxes for the current exported PNG; no camera, controller or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

ROLES=('white_mug','red_mug','box')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--boxes',help='Manually supplied JSON role -> [x0,y0,x1,y1]; no automatic inference')
    a=p.parse_args(argv);source=a.package/'local/current_first_frame.png';raw=source.read_bytes()
    import cv2
    import numpy as np
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('current first-frame PNG cannot be decoded')
    height,width=image.shape[:2]
    if a.boxes:boxes=json.loads(a.boxes)
    else:
        boxes={}
        try:
            for role in ROLES:
                x,y,w,h=cv2.selectROI('Current first frame: '+role,image,showCrosshair=True,fromCenter=False)
                boxes[role]=[int(x),int(y),int(x+w),int(y+h)]
                cv2.destroyAllWindows()
        finally:cv2.destroyAllWindows()
    if not isinstance(boxes,dict) or set(boxes)!=set(ROLES):raise ValueError('exactly white_mug, red_mug and box must be manually supplied')
    for role,box in boxes.items():
        if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) for v in box):raise ValueError('invalid box: '+role)
        x0,y0,x1,y1=box
        if not (0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('box is empty or outside the current image: '+role)
    payload={'frame_sha256':hashlib.sha256(raw).hexdigest(),'role_boxes':boxes}
    destination=a.package/'local/current_first_frame_boxes.json'
    temp=destination.with_name('.'+destination.name+'.'+uuid.uuid4().hex)
    with temp.open('x') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,destination);print(destination);return payload

if __name__=='__main__':main()
