# minimal comments
import pandas as pd
from typing import Dict, Set, List

def build_report(room2classes: Dict[str, Set[str]], class_names_ordered: List[str]):
    cols = [c.capitalize() for c in class_names_ordered]
    rows = []
    idx = []
    for room, clsset in room2classes.items():
        row = [1 if c.lower() in clsset else 0 for c in class_names_ordered]
        rows.append(row)
        idx.append(room)
    df = pd.DataFrame(rows, index=idx, columns=cols)
    # add Total row
    if not df.empty:
        total = df.sum(axis=0)
        total.name = "Total"
        df = pd.concat([df, pd.DataFrame([total], index=["Total"])])
    return df
