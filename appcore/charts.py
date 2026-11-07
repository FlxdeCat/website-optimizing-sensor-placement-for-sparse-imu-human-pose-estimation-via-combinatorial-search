import numpy as np
import plotly.graph_objects as go
from .constants import PARENTS,POOL,NAMES,NAMES_ID
COLORS=['#bf6849','#4f8072','#7088a2','#ba9a58','#a67b91','#7c7e73','#363d48']
def style(fig,dark=False,height=400):
    fg='#eee9e0' if dark else '#292d28';bg='#181d1a' if dark else '#faf8f2'
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white',paper_bgcolor=bg,plot_bgcolor=bg,font=dict(family='Arial, sans-serif',color=fg,size=13),margin=dict(l=15,r=15,t=35,b=35),height=height,legend=dict(orientation='h',y=-.18,x=0),hoverlabel=dict(font_size=13))
    return fig
def body(positions,selected,dark=False,lang='en',prediction=None):
    fig=go.Figure();names=NAMES_ID if lang=='id' else NAMES
    def add(p,color,name):
        # Plotly Z is vertical. Original coordinates use Y up.
        p=np.asarray(p)[:,[0,2,1]];x=[];y=[];z=[]
        for j,parent in enumerate(PARENTS):
            if parent>=0:
                x.extend([p[parent,0],p[j,0],None]);y.extend([p[parent,1],p[j,1],None]);z.extend([p[parent,2],p[j,2],None])
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color=color,width=6),name=name,hoverinfo='skip'))
        return p
    p=add(positions,'#a6aaa1','Reference' if lang=='en' else 'Acuan')
    if prediction is not None:add(prediction,'#bf6849','Prediction' if lang=='en' else 'Prediksi')
    rest=[j for j in POOL if j not in selected]
    for group,color,size,name in [(rest,'#8b938a',4,'Available' if lang=='en' else 'Tersedia'),([j for j in selected if j!=0],'#bf6849',8,'Selected' if lang=='en' else 'Terpilih'),([0],'#4f8072',10,'Pelvis · required' if lang=='en' else 'Panggul · wajib')]:
        if group:fig.add_trace(go.Scatter3d(x=p[group,0],y=p[group,1],z=p[group,2],mode='markers',marker=dict(color=color,size=size,line=dict(color='#faf8f2',width=1)),text=[f'{j}: {names[j]}' for j in group],hovertemplate='%{text}<extra></extra>',name=name))
    fig.update_layout(scene=dict(aspectmode='cube',xaxis=dict(visible=False,range=[-.95,.95]),yaxis=dict(visible=False,range=[-.95,.95]),zaxis=dict(visible=False,range=[-.95,.95]),camera=dict(eye=dict(x=.55,y=1.23,z=.25)),bgcolor='#181d1a' if dark else '#faf8f2'),uirevision='body-camera')
    return style(fig,dark,490)
