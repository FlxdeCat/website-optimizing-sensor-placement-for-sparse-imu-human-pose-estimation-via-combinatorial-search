"""Client-side playback: no Streamlit reruns or inference during frame changes."""
import json
from functools import lru_cache
import numpy as np
import streamlit.components.v1 as components
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder
from .charts import body
from .constants import POOL,PARENTS
@lru_cache(maxsize=1)
def plotly_js():return get_plotlyjs()
def playback_html(reference,prediction,selected,errors,dark=False,lang='en'):
    reference=np.asarray(reference);prediction=np.asarray(prediction)
    if reference.shape!=prediction.shape or reference.shape[1:]!=(24,3) or len(errors)!=len(reference):raise ValueError('Playback arrays must match [T,24,3]')
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():raise ValueError('Playback positions must be finite')
    selected=list(selected);rest=[j for j in POOL if j not in selected];chosen=[j for j in selected if j!=0]
    def lines(p):
        p=p[:,[0,2,1]];out=[[],[],[]]
        for j,parent in enumerate(PARENTS):
            if parent>=0:
                for axis in range(3):out[axis].extend([float(p[parent,axis]),float(p[j,axis]),None])
        return out
    frames=[]
    for ref,pred in zip(reference,prediction):
        row=[lines(ref),lines(pred)];p=ref[:,[0,2,1]]
        for group in [rest,chosen,[0]]:
            if group:row.append([p[group,axis].tolist() for axis in range(3)])
        frames.append(row)
    fig=body(reference[0],selected,dark,lang,prediction[0]);fig.update_layout(height=None,autosize=True,margin=dict(l=0,r=0,t=5,b=25),legend=dict(y=0,x=0,bgcolor='#181d1a' if dark else '#faf8f2'))
    payload=json.dumps({'figure':fig.to_plotly_json(),'frames':frames,'errors':np.asarray(errors).tolist()},cls=PlotlyJSONEncoder,allow_nan=False)
    fg,bg,border=('#eee9e0','#181d1a','#465047') if dark else ('#292d28','#faf8f2','#d4d7cc')
    labels=['Play','Stop','Previous frame','Next frame','Frame','Schematic MPJPE'] if lang=='en' else ['Putar','Berhenti','Frame sebelumnya','Frame berikutnya','Frame','MPJPE skematis']
    # All labels are fixed application strings; data is JSON, never executable user input.
    template='''<!doctype html><html lang="__LANG__"><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>
    *{box-sizing:border-box}body{margin:0;font:14px Arial,sans-serif;background:__BG__;color:__FG__}#plot{width:100%;height:465px;overflow:hidden;position:relative}.controls{position:relative;z-index:5;display:flex;gap:8px;flex-wrap:wrap;padding:8px 0}button{min-height:44px;padding:8px 13px;border:1px solid __BORDER__;border-radius:4px;background:__BG__;color:__FG__;font:inherit;cursor:pointer}button:hover{border-color:#c58162}button[aria-pressed=true]{background:#ad5437;color:white}input{width:100%;margin:0;accent-color:#ad5437;min-height:30px}#readout{line-height:1.5;padding:4px 0}button:focus-visible,input:focus-visible{outline:2px solid #bf6849;outline-offset:2px}@media(max-width:600px){#plot{height:390px}.controls{gap:6px}button{flex:1 1 40%;font-size:13px;padding:8px}}
    </style><script>__PLOTLY__</script></head><body><div class="controls"><button id="play">▶ __PLAY__</button><button id="stop">■ __STOP__</button><button id="prev">← __PREV__</button><button id="next">__NEXT__ →</button></div><div id="plot"></div>
    <input id="scrub" type="range" min="0" max="__MAX__" value="0" aria-label="__FRAME__"><div id="readout" aria-live="off"></div>
    <script>
    const data=__DATA__,g=document.getElementById('plot'),scrub=document.getElementById('scrub'),readout=document.getElementById('readout');
    let frame=0,playing=false,raf=0,start=0,startFrame=0,lastDraw=0,ready=false,busy=false;
    function status(){scrub.value=frame;readout.textContent='__FRAME__ '+frame+' / '+(data.frames.length-1)+' · '+(frame/60).toFixed(2)+' s · __METRIC__: '+data.errors[frame].toFixed(3)+' cm';}
    async function draw(next){frame=Math.max(0,Math.min(data.frames.length-1,next));status();if(!ready||busy)return;busy=true;const drawn=frame,row=data.frames[frame];try{await Plotly.restyle(g,{x:row.map(r=>r[0]),y:row.map(r=>r[1]),z:row.map(r=>r[2])},row.map((r,i)=>i));}finally{busy=false;if(frame!==drawn)draw(frame);}}
    function stop(){playing=false;cancelAnimationFrame(raf);document.getElementById('play').setAttribute('aria-pressed','false');}
    function tick(now){if(!playing)return;let target=Math.min(data.frames.length-1,startFrame+Math.floor((now-start)*.06));if(now-lastDraw>65||target===data.frames.length-1){lastDraw=now;draw(target);}if(target===data.frames.length-1){stop();return;}raf=requestAnimationFrame(tick);}
    document.getElementById('play').onclick=()=>{if(!ready||playing)return;if(frame===data.frames.length-1)frame=0;playing=true;start=performance.now();startFrame=frame;document.getElementById('play').setAttribute('aria-pressed','true');raf=requestAnimationFrame(tick);};
    document.getElementById('stop').onclick=stop;
    document.getElementById('prev').onclick=()=>{stop();draw(frame-1);};document.getElementById('next').onclick=()=>{stop();draw(frame+1);};scrub.oninput=()=>{stop();draw(Number(scrub.value));};
    Plotly.newPlot(g,data.figure.data,data.figure.layout,{responsive:true,displaylogo:false,scrollZoom:true}).then(()=>{ready=true;status();});
    window.addEventListener('resize',()=>{if(ready)Plotly.Plots.resize(g);});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});status();
    </script></body></html>'''
    replacements={'__LANG__':lang,'__BG__':bg,'__FG__':fg,'__BORDER__':border,'__PLOTLY__':plotly_js(),'__PLAY__':labels[0],'__STOP__':labels[1],'__PREV__':labels[2],'__NEXT__':labels[3],'__FRAME__':labels[4],'__METRIC__':labels[5],'__MAX__':str(len(reference)-1),'__DATA__':payload}
    for key,value in replacements.items():template=template.replace(key,value)
    return template
def playback(reference,prediction,selected,errors,dark=False,lang='en'):
    components.html(playback_html(reference,prediction,selected,errors,dark,lang),height=610,scrolling=False)
