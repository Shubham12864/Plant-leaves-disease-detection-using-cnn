"""Create train/validation partitions from an already hashed source-train manifest."""
from __future__ import annotations
import argparse, csv, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--train-ratio',type=float,default=.85)
    args=ap.parse_args()
    if not 0 < args.train_ratio < 1: ap.error('train-ratio must be between 0 and 1')
    with args.input.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f); fields=list(reader.fieldnames or []); rows=list(reader)
    if not rows: ap.error('input manifest is empty')
    if 'group_id' not in fields or any(not r.get('group_id') for r in rows): ap.error('all rows need a stable group_id')
    # Union records that share a physical group or exact image bytes.
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b):
        x,y=find(a),find(b)
        if x!=y: parent[max(x,y)]=min(x,y)
    seen={}
    for i,row in enumerate(rows):
        for key in (('group',row['group_id']),('hash',row.get('sha256',''))):
            if key[1]:
                if key in seen: union(i,seen[key])
                else: seen[key]=i
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[find(i)].append(row)
    names=list(groups); random.Random(args.seed).shuffle(names)
    target=args.train_ratio*len(rows); train_count=0
    for key in sorted(names,key=lambda k:-len(groups[k])):
        split='train' if train_count < target else 'validation'
        for row in groups[key]: row['split']=split
        if split=='train': train_count+=len(groups[key])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    counts=Counter(r['split'] for r in rows)
    per_split={split:dict(sorted(Counter(r['label'] for r in rows if r['split']==split).items())) for split in ('train','validation')}
    summary={'source_manifest':str(args.input),'seed':args.seed,'group_preserving':True,'train_ratio_target':args.train_ratio,'rows':len(rows),'splits':dict(counts),'classes':per_split,'warning':'Greedy group-size assignment is not class-stratified; inspect per-class support before training.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'rows':len(rows),'splits':dict(counts),'groups':len(groups)},indent=2))

if __name__=='__main__': main()
