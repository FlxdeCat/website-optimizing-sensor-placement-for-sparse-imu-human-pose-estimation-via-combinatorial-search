"""Independent PyTorch implementation of the audited Linear/LSTM/Linear graph.
No upstream implementation is distributed. CPU FP32, window 60, stride 60.
"""
import hashlib,logging
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from .constants import ROOT,ALIASES,PARENTS
from .data import manifest
from .kinematics import fk
LOGGER=logging.getLogger(__name__)
class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(288,64);self.lstm=nn.LSTM(64,64,2,bidirectional=True,batch_first=True);self.linear2=nn.Linear(128,216)
    def forward(self,x):return self.linear2(self.lstm(self.linear(x))[0])
class PoseModel(nn.Module):
    def __init__(self):super().__init__();self.backbone=Backbone()
    def forward(self,x):return self.backbone(x).reshape(x.shape[0],x.shape[1],24,3,3)
def pack(acc,ori,joints):
    joints=tuple(sorted(joints));acc=np.asarray(acc,dtype=np.float32);ori=np.asarray(ori,dtype=np.float32)
    if 0 not in joints or len(set(joints))!=len(joints):raise ValueError('Distinct sites including pelvis required')
    if acc.ndim!=3 or acc.shape[1:]!=(24,3) or ori.shape!=(len(acc),24,3,3):raise ValueError('Invalid input shapes')
    if not np.isfinite(acc).all() or not np.isfinite(ori).all():raise ValueError('Nonfinite signals')
    a=np.zeros_like(acc);o=np.zeros_like(ori);root=ori[:,0].transpose(0,2,1)
    for j in joints:
        v=acc[:,j] if j==0 else acc[:,j]-acc[:,0]
        a[:,j]=np.einsum('tij,tj->ti',root,v)/30
        o[:,j]=ori[:,j] if j==0 else root@ori[:,j]
    # All 72 acceleration channels first; then all 216 orientation channels.
    return np.concatenate((a.reshape(len(acc),72),o.reshape(len(acc),216)),axis=1)
def project(raw):
    u,s,vh=np.linalg.svd(np.asarray(raw,dtype=np.float64));sign=np.ones(s.shape);sign[...,-1]=np.where(np.linalg.det(u@vh)<0,-1,1)
    return (u*sign[...,None,:])@vh
@lru_cache(maxsize=5)
def load_model(label):
    label=ALIASES.get(label,label);path=ROOT/'models'/(label+'.pt')
    if not path.exists():raise FileNotFoundError(f'Model file missing: {path.name}')
    entry=manifest()[label]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['weights_sha256']:raise ValueError('Model checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['key']!=entry['key'] or payload['epoch']!=5:raise ValueError('Wrong checkpoint identity or budget')
    torch.set_num_threads(1);model=PoseModel();model.load_state_dict(payload['model'],strict=True);model.eval()
    LOGGER.info('Loaded %s (seed 1234, final epoch 5)',label);return model
@torch.inference_mode()
def predict(label,acc,ori):
    label=ALIASES.get(label,label);model=load_model(label);x=pack(acc,ori,manifest()[label]['joints']);parts=[]
    for start in range(0,len(x),60):parts.append(model(torch.from_numpy(x[start:start+60])[None])[0].numpy())
    return project(np.concatenate(parts))
def angles(a,b):
    r=np.swapaxes(a,-1,-2)@b;skew=np.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),-1)
    return np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1)/2,np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1)))
def demo_metrics(pred,truth):
    pg,pp=fk(pred);tg,tp=fk(truth);local=angles(pred,truth);glob=angles(pg,tg);error=np.linalg.norm(pp-tp,axis=-1)*100
    jerk=np.diff(pp-pp[:,:1],n=3,axis=0)*60**3/1000
    return {'local_rotation_deg':float(local.mean()),'global_rotation_deg':float(glob.mean()),'sip_deg':float(glob[:,[1,2,16,17]].mean()),'mpjpe_cm':float(error.mean()),'mesh_cm':None,'jitter_km_s3':float(np.linalg.norm(jerk,axis=-1).mean()),'per_frame_mpjpe':error.mean(axis=1),'pred_positions':pp,'ref_positions':tp}
