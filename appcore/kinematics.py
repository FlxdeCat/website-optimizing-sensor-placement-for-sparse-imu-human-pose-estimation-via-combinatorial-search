"""Hand-authored schematic body, not SMPL geometry or dataset-derived assets."""
import numpy as np
from .constants import PARENTS
# Offsets in metres in a right-handed, Y-up schematic body.
OFFSETS=np.array([[0,0,0],[.09,-.08,0],[-.09,-.08,0],[0,.12,0],[0,-.40,0],[0,-.40,0],[0,.12,0],[0,-.39,0],[0,-.39,0],[0,.15,0],[0,-.04,.13],[0,-.04,.13],[0,.15,0],[.09,.08,0],[-.09,.08,0],[0,.14,0],[.10,0,0],[-.10,0,0],[.28,0,0],[-.28,0,0],[.25,0,0],[-.25,0,0],[.08,0,0],[-.08,0,0]],dtype=float)
def rotation(axis,angle):
    a=np.asarray(angle);c=np.cos(a);s=np.sin(a);r=np.broadcast_to(np.eye(3),a.shape+(3,3)).copy();i,j=((1,2),(2,0),(0,1))[axis];r[...,i,i]=c;r[...,j,j]=c;r[...,i,j]=-s;r[...,j,i]=s;return r
def fk(local):
    local=np.asarray(local);shape=local.shape[:-3];g=np.empty_like(local);pos=np.zeros(shape+(24,3))
    for j,p in enumerate(PARENTS):
        if p<0:g[...,j,:,:]=local[...,j,:,:]
        else:
            g[...,j,:,:]=g[...,p,:,:]@local[...,j,:,:]
            pos[...,j,:]=pos[...,p,:]+np.einsum('...ij,j->...i',g[...,p,:,:],OFFSETS[j])
    return g,pos
def generate(motion='walk',seconds=4,speed=1.,amplitude=1.):
    t=np.arange(int(seconds*60))/60;wave=np.sin(2*np.pi*speed*t);local=np.broadcast_to(np.eye(3),(len(t),24,3,3)).copy()
    if motion=='walk':
        for j,sign in [(1,1),(2,-1),(16,-1),(17,1)]:local[:,j]=rotation(0,sign*.45*amplitude*wave)
        for j,sign in [(4,1),(5,-1)]:local[:,j]=rotation(0,.5*amplitude*np.maximum(sign*wave,0))
    elif motion=='reach':
        local[:,16]=rotation(2,-.65*amplitude*(1+wave));local[:,18]=rotation(1,.45*amplitude*wave)
    elif motion=='bend':local[:,3]=rotation(0,.35*amplitude*(1+wave));local[:,6]=rotation(0,.25*amplitude*(1+wave))
    else:raise ValueError('Unknown generated motion')
    ori,pos=fk(local);acc=np.zeros_like(pos)
    acc[1:-1]=(pos[2:]+pos[:-2]-2*pos[1:-1])*3600
    acc[4:-4]=(pos[8:]+pos[:-8]-2*pos[4:-4])*(3600/16)
    return {'pose':local,'ori':ori,'acc':acc,'positions':pos,'fps':60}
