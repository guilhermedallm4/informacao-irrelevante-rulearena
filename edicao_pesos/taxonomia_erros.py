import json, sys, re, collections
sys.path.insert(0,'.')
from src.airline.gold import load_airline_problems, get_check_base_tables, compute_oversize, compute_overweight
rows=[json.loads(l) for l in open(sys.argv[1])]
clean=[r for r in rows if r['scenario']=='clean']
probs={c:load_airline_problems(c) for c in (0,1,2)}
tables=get_check_base_tables()
cnt=collections.Counter(); per_prob=collections.Counter()
item_re=re.compile(r"Item (\d+) \(.*?\):\*?\*?\n(.*?)(?=\n\d+\. \*?\*?Item |\n#+ ?Summary|\nSummary)", re.S)
for r in clean:
    c=int(r['instance_id'][1]); idx=int(r['instance_id'][3:])
    info=probs[c][idx]['info']; bags=info['bag_list'][1:]
    base=[int(tables[min(3,i)][info['direction']][info['customer_class']][info['routine']]) for i in range(len(bags))]
    kinds=set()
    for m in item_re.finditer(r['response']):
        n=int(m.group(1)); body=m.group(2)
        if n-2>=len(bags): continue
        b=bags[n-2]; k=n-2
        sz=sum(b['size']); w=b['weight']
        g_os=compute_oversize(b,info['routine']); g_ow=compute_overweight(b,info['routine'],info['customer_class'],base[k]==0)
        mc=re.search(r"checking fee is \$(\d+)",body); m_os=re.search(r"Oversize Fee:\n\s*- .*?\$(\d+)",body); m_ow=re.search(r"Overweight Fee:\n\s*- .*?\$(\d+)",body)
        tot=re.search(r"Total Fee for Item \d+: .*?= \$([\d,]+)",body)
        p_chk=int(mc.group(1)) if mc else None
        p_os=int(m_os.group(1)) if m_os else (0 if re.search(r"Oversize Fee:.*?\$0|No oversize|Within the standard size",body,re.S) else None)
        p_ow=int(m_ow.group(1)) if m_ow else (0 if re.search(r"Overweight Fee:.*?\$0|No overweight|Within the standard weight",body,re.S) else None)
        if p_chk is not None and p_chk!=base[k]: kinds.add(f"checking bag{k+1}: {p_chk} vs {base[k]}")
        if p_os is not None and p_os!=g_os:
            tier='63-65' if 62<sz<=65 else ('>65' if sz>65 else '<=62')
            kinds.add(f"oversize tier {tier}: {p_os} vs {g_os}")
        if p_ow is not None and p_ow!=g_ow:
            tier='51-53' if 50<w<=53 else ('54-70' if 53<w<=70 else ('>70' if w>70 else '<=50'))
            comp=' (comp)' if base[k]==0 and info['customer_class'] in ('Business','First') else ''
            kinds.add(f"overweight tier {tier}{comp}: {p_ow} vs {g_ow}")
        if tot and p_chk is not None and p_os is not None and p_ow is not None:
            t=int(tot.group(1).replace(',',''))
            if t!=p_chk+max(p_os,p_ow): kinds.add("soma do item != chk+max(os,ow)")
    for kk in kinds: cnt[kk.split(':')[0]]+=1
    print(r['instance_id'], 'OK' if r['correct'] else f"diff={r['pred_cents']/100-r['gold'] if r['pred_cents'] else None}", sorted(kinds))
print()
for k,v in cnt.most_common(): print(f"{v:3d} problemas: {k}")
