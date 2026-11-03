import numpy as np
import sys
C = sys.argv[1].rstrip('/') + '/'
cov=np.load('results/coco/per_image/covariates.npz'); size=cov['mean_obj_size']/256
qi=np.searchsorted(np.quantile(size,[.2,.4,.6,.8]),size)
PAIRS=[('trm_t1','trm'),('trm_t1_seed1','trm_seed1'),('trm_t1_seed2','trm_seed2')]
def rates(d,t):
    g=lambda k:d[k][:,t].astype(float)
    W,Wa,SC,SCa,DC,DCa=(g(k) for k in ('W','Wa','SC','SCa','DC','DCa'))
    T=Wa+SCa+DCa
    with np.errstate(invalid='ignore',divide='ignore'):
        return dict(fg_ari=g('fg_ari'),mbo_i=g('mbo_i'),mbo_c=g('mbo_c'),max_share=g('max_share'),
            w=W/Wa, q=(W+SC+DC)/T, r_sc=np.where(SCa>0,SC/SCa,np.nan), r_dc=np.where(DCa>0,DC/DCa,np.nan),
            w_c=(W+SC)/(Wa+SCa), cW=W/T, cSC=SC/T, cDC=DC/T)
sel={'all':np.ones(len(qi),bool),'smallest':qi==0,'largest':qi==4}
both=None
A=[dict(np.load(C+f'p_{a}.npz')) for a,_ in PAIRS]; B=[dict(np.load(C+f'p_{b}.npz')) for _,b in PAIRS]
# images with both same-class and different-class object pairs, so r_sc and r_dc are both defined
ok=np.all([(x['SCa'][:,0]>0)&(x['DCa'][:,0]>0) for x in A],0)
print('images with both same-class and different-class object pairs: %d of %d'%(ok.sum(),len(ok)), ' by quintile', [int(ok[qi==k].sum()) for k in range(5)])
def mean(ds,t,k,s,need_ok=False):
    m=sel[s]&(ok if need_ok else True)
    return np.mean([np.nanmean(rates(d,t)[k][m]) for d in ds])
print('\n=== per recursion step (T=8 models, mean of 3 seeds), with the T=1 model for reference ===')
for s in ('all','smallest','largest'):
    print(f'-- {s} images')
    for k,sc in (('fg_ari',100),('mbo_i',100),('mbo_c',100),('w',1),('q',1),('r_sc',1),('r_dc',1),('max_share',100)):
        need=k in('r_sc','r_dc')
        print(f'   {k:9s} T=1 model {sc*mean(A,0,k,s,need):7.3f} | T=8 steps '+' '.join('%7.3f'%(sc*mean(B,t,k,s,need)) for t in range(8)))
print('\n=== T=1 -> T=8 (final step), by object-size quintile small -> large ===')
def prof(ds,t,k,need=False):
    return np.mean([[np.nanmean(rates(d,t)[k][(qi==q)&(ok if need else True)]) for q in range(5)] for d in ds],0)
for k in ('w','r_sc','r_dc','w_c'):
    need=k in('r_sc','r_dc')
    a,b=prof(A,0,k,need),prof(B,7,k,need)
    print(f'{k:5s} T=1 '+' '.join('%.3f'%x for x in a)+'   T=8 '+' '.join('%.3f'%x for x in b)+'   rel change '+' '.join('%+5.1f%%'%(100*(y/x-1)) for x,y in zip(a,b)))
print('\nchange in q split by pair type (sums to Δq):')
for k,lab in (('cW','within-object pairs'),('cSC','same-class pairs  '),('cDC','diff-class pairs  ')):
    d=prof(B,7,k)-prof(A,0,k)
    print(f'  {lab} '+' '.join('%+.4f'%x for x in d))
print('  total Δq           '+' '.join('%+.4f'%x for x in prof(B,7,'q')-prof(A,0,'q')))
print('\nshare of fg pairs by type (T=1 model): within / same-class / diff-class')
for q in range(5):
    m=qi==q
    W=np.mean([np.nanmean((d['Wa'][:,0]/(d['Wa'][:,0]+d['SCa'][:,0]+d['DCa'][:,0]))[m]) for d in A])
    S=np.mean([np.nanmean((d['SCa'][:,0]/(d['Wa'][:,0]+d['SCa'][:,0]+d['DCa'][:,0]))[m]) for d in A])
    print(f'  quintile {q}: {W:.3f} / {S:.3f} / {1-W-S:.3f}')
