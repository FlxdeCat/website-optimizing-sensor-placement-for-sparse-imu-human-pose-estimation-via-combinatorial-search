"""Fixed public SMPL-Body asset. No shape space, full SMPL model or private data.
The display uses equal world-unit scale on every axis; mesh geometry is unaltered.
Illustrative markers are nearest surface vertices to hand-set anatomical targets.
They are not the synthesis pipeline's joint-regressor argmax vertices.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go
from .constants import ROOT,POOL,NAMES,NAMES_ID
from .charts import style
@lru_cache(maxsize=1)
def asset():
    with np.load(ROOT/'assets/smpl_body/body.npz',allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}
def placement(selected,dark=False,lang='en',view='three_quarter'):
    cameras={'three_quarter':dict(x=1.2,y=2.05,z=.2),'front':dict(x=0,y=2.3,z=.15),'side':dict(x=2.35,y=.05,z=.15)}
    if view not in cameras:raise ValueError('Unknown viewing angle')
    a=asset();v=a['vertices'][:,[0,2,1]];f=a['faces'];points=a['markers'][:,[0,2,1]];names=NAMES if lang=='en' else NAMES_ID
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color='#aeb6ac' if dark else '#b8bcb4',flatshading=False,lighting=dict(ambient=.55,diffuse=.75,specular=.12,roughness=.8),lightposition=dict(x=2,y=4,z=5),hoverinfo='skip',showlegend=False))
    for indices,color,size,name in [([n for n,j in enumerate(POOL) if j not in selected],'#788a9b',4,'Available' if lang=='en' else 'Tersedia'),([n for n,j in enumerate(POOL) if j in selected and j!=0],'#d7774f',7,'Selected' if lang=='en' else 'Terpilih'),([0],'#4f9b7f',9,'Pelvis · required' if lang=='en' else 'Panggul · wajib')]:
        if not indices:continue
        p=points[indices];fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(color=color,size=size,line=dict(color='#eee9df',width=1)),name=name,text=[f'{POOL[n]}: {names[POOL[n]]}' for n in indices],hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(scene=dict(aspectmode='manual',aspectratio=dict(x=1.04,y=.52,z=1.86),xaxis=dict(visible=False,range=[-.52,.52]),yaxis=dict(visible=False,range=[-.24,.28]),zaxis=dict(visible=False,range=[-.03,1.83]),camera=dict(eye=cameras[view],projection=dict(type='perspective')),bgcolor='#181d1a' if dark else '#faf8f2'),uirevision='smpl-placement-'+view)
    style(fig,dark,580);fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),legend=dict(y=0,x=0,bgcolor='#181d1a' if dark else '#faf8f2'))
    return fig
