# minimal comments
from typing import List, Set, Dict

def dedupe_per_room(detections: List[Dict], class_names: Dict[int, str], conf_thresh: float = 0.25) -> Set[str]:
    s = set()
    for d in detections:
        if d["conf"] < conf_thresh:
            continue
        cid = d["class_id"]
        name = class_names.get(cid, str(cid)).lower()
        s.add(name)
    return s
