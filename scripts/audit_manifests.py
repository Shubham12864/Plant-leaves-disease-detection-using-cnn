"""Summarize class coverage and exact/group leakage across manifest partitions."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifests',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    rows=[]
    for p in args.manifests:
        if p.exists():
            for row in read_rows(p): row['_manifest']=p.name; rows.append(row)
    if not rows: ap.error('no manifest rows found')
    summary={'rows':len(rows),'manifests':{},'exact_hash_overlap':{},'group_overlap':{},'class_intersection':{}}
    per_ds=defaultdict(list)
    for row in rows: per_ds[row['dataset']].append(row)
    for ds,group in per_ds.items():
        split_counts=defaultdict(list)
        for r in group: split_counts[r.get('split','')].append(r)
        summary['manifests'][ds]={s:{'images':len(items),'classes':dict(sorted(Counter(x['label'] for x in items).items())),'unreadable':0} for s,items in split_counts.items()}
        if 'train' in split_counts and 'test' in split_counts:
            tr=split_counts['train']+split_counts.get('validation',[])
            te=split_counts['test']
            htr=defaultdict(list); hte=defaultdict(list)
            for r in tr: htr[r.get('sha256','')].append(r['relative_path'])
            for r in te: hte[r.get('sha256','')].append(r['relative_path'])
            overlap=sorted((set(htr)&set(hte))-{''})
            summary['exact_hash_overlap'][ds]={'count':len(overlap),'examples':[{'sha256':h,'train':htr[h][:3],'test':hte[h][:3]} for h in overlap[:20]]}
            gtr={r.get('group_id','') for r in tr if r.get('group_id')}
            gte={r.get('group_id','') for r in te if r.get('group_id')}
            go=sorted(gtr&gte)
            summary['group_overlap'][ds]={'count':len(go),'examples':go[:20]}
    names=list(per_ds)
    if len(names)>=2:
        classes={ds:{r['label'] for r in group} for ds,group in per_ds.items()}
        for i,a in enumerate(names):
            for b in names[i+1:]: summary['class_intersection'][f'{a}__{b}']=sorted(classes[a]&classes[b])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'rows':len(rows),'datasets':{k:{s:v['images'] for s,v in splits.items()} for k,splits in summary['manifests'].items()},'exact_hash_overlap':{k:v['count'] for k,v in summary['exact_hash_overlap'].items()},'group_overlap':{k:v['count'] for k,v in summary['group_overlap'].items()},'class_intersection_sizes':{k:len(v) for k,v in summary['class_intersection'].items()}},indent=2))

if __name__=='__main__': main()
