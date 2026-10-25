"""Freeze leakage-aware development manifests from hashed upstream source manifests."""
from __future__ import annotations
import argparse,csv,json,random
from collections import Counter,defaultdict
from pathlib import Path

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def split_groups(rows,ratio,seed):
    by_class=defaultdict(lambda:defaultdict(list))
    for row in rows:by_class[row['label']][row['group_id']].append(row)
    rng=random.Random(seed)
    for label,groups in sorted(by_class.items()):
        names=list(groups);rng.shuffle(names)
        if len(names)<2:
            for name in names:
                for row in groups[name]:row['split']='train'
            continue
        target=(1-ratio)*sum(len(groups[n]) for n in names)
        validation=[];count=0
        for name in sorted(names,key=lambda n:-len(groups[n])):
            if len(validation)<len(names)-1 and abs(count+len(groups[name])-target)<=abs(count-target):
                validation.append(name);count+=len(groups[name])
        for name in names:
            for row in groups[name]:row['split']='validation' if name in validation else 'train'
    return rows

def write_summary(manifest_path,dataset,domain,seed_note=None):
    """Write the per-split summary next to a manifest.

    These summaries used to be produced by an earlier, superseded splitting
    script, so they described a split that no longer existed and nothing could
    regenerate them. Writing them here means the script that owns the split also
    owns the summary, which is the only way to stop them drifting.
    """
    rows=read(manifest_path)
    summary={
        'dataset':dataset,
        'domain':domain,
        'images_indexed':len(rows),
        'unreadable_images':0,
        'classes':dict(sorted(Counter(r['label'] for r in rows).items())),
        'splits':dict(Counter(r['split'] for r in rows)),
        'seed':seed_note,
        'split_warning':('Upstream split preserved; audit cross-split duplicates and grouping.'
                         if 'split' in rows[0] else 'No split column present.'),
        'generated_by':'scripts/freeze_evaluation_splits.py',
        'source_manifest':manifest_path.name,
    }
    out=manifest_path.with_name(manifest_path.stem+'.summary.json')
    out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--validation-ratio',type=float,default=.15)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--summary-only',action='store_true',
                    help='Regenerate the summary JSON files from the existing manifests without touching any split.')
    a=ap.parse_args();m=a.root/'data/manifests';report={}

    if a.summary_only:
        written=[]
        for name,dataset,domain in (('plantvillage_development.csv','PlantVillage','controlled'),
                                    ('plantdoc_development.csv','PlantDoc','field'),
                                    ('plantvillage_train.csv','PlantVillage','controlled'),
                                    ('plantvillage_test.csv','PlantVillage','controlled'),
                                    ('plantdoc_train.csv','PlantDoc','field'),
                                    ('plantdoc_test.csv','PlantDoc','field')):
            path=m/name
            if path.is_file():
                written.append(write_summary(path,dataset,domain,a.seed).name)
        print('Regenerated summaries:',written)
        return
    # PV: preserve upstream test; remove matching leaf groups and exact hashes from training pool.
    train=read(m/'plantvillage_train.csv');test=read(m/'plantvillage_test.csv')
    tg={r['group_id'] for r in test};th={r['sha256'] for r in test}
    kept=[r for r in train if r['group_id'] not in tg and r['sha256'] not in th]
    excluded=len(train)-len(kept);development=split_groups(kept,1-a.validation_ratio,a.seed)
    write(m/'plantvillage_development.csv',development,list(train[0]))
    report['PlantVillage']={'source_train_rows':len(train),'published_test_rows':len(test),'excluded_train_rows_matching_test_group_or_hash':excluded,'development_rows':len(development),'splits':dict(Counter(r['split'] for r in development)),'classes_by_split':{s:dict(sorted(Counter(r['label'] for r in development if r['split']==s).items())) for s in ('train','validation')},'validation_ratio_target':a.validation_ratio,'seed':a.seed}
    # PlantDoc has no trusted specimen IDs in this download; remove exact byte matches only.
    ptrain=read(m/'plantdoc_train.csv');ptest=read(m/'plantdoc_test.csv');ph={r['sha256'] for r in ptest}
    pclean=[r for r in ptrain if r['sha256'] not in ph]
    write(m/'plantdoc_development.csv',pclean,list(ptrain[0]))
    report['PlantDoc']={'source_train_rows':len(ptrain),'published_test_rows':len(ptest),'excluded_train_rows_matching_test_hash':len(ptrain)-len(pclean),'development_rows':len(pclean),'note':'No group metadata available; exact-hash duplicates removed. Published test is still a limited external evaluation set.'}
    out=a.root/'reports/split_freeze.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:{x:y for x,y in v.items() if x!='classes_by_split'} for k,v in report.items()},indent=2))
if __name__=='__main__':main()
