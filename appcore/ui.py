"""Explicitly themed semantic HTML tables: no light-only canvas grid."""
import html
import numpy as np
import pandas as pd
import streamlit as st
def numbered_table(data,**kwargs):
    frame=pd.DataFrame(data).copy().reset_index(drop=True)
    if '#' not in frame:frame.insert(0,'#',range(1,len(frame)+1))
    # Escaping remains enabled, including source labels.
    markup=frame.to_html(index=False,border=0,classes='research-table',escape=True,float_format=lambda x:f'{x:.4f}',na_rep='N/A')
    st.markdown('<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable research table">'+markup+'</div>',unsafe_allow_html=True)
def theme_css(dark):
    fg,bg,muted,border,surface=('#eee9e0','#181d1a','#bbc2b7','#465047','#232b24') if dark else ('#292d28','#faf8f2','#657064','#d4d7cc','#eeeee5')
    return f'''<style>
:root{{color-scheme:{'dark' if dark else 'light'};--text-color:{fg};--background-color:{bg};--secondary-background-color:{surface};}}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{{background:{bg}!important;color:{fg}!important;}}
[data-testid="stSidebar"]{{background:{surface}!important;color:{fg}!important;border-right:1px solid {border};}}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *,[data-testid="stMetricLabel"],[data-testid="stMetricDelta"],label,[data-testid="stWidgetLabel"],.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp li{{color:{fg}!important;}}
[data-testid="stCaptionContainer"] p{{color:{muted}!important;}}
.stButton button,.stDownloadButton button,[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-tertiary"]{{background:{surface}!important;color:{fg}!important;border:1px solid {border}!important;border-radius:4px;}}
.stButton button p,.stDownloadButton button p{{color:inherit!important;}}
.stButton button[kind="primary"]{{background:#ad5437!important;color:#fff!important;border-color:#ad5437!important;}}
.stButton button[kind="primary"] p{{color:#fff!important;}}
.stButton button:hover,.stDownloadButton button:hover{{border-color:#c58162!important;}}
[data-baseweb="select"]>div,[data-baseweb="select"] span,[data-baseweb="input"],input,textarea,[role="listbox"],[role="option"],[data-baseweb="popover"]{{background:{surface}!important;color:{fg}!important;}}
[data-baseweb="select"] svg{{fill:{fg}!important;}}
[data-testid="stSlider"] [data-testid="stThumbValue"],[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"]{{color:{fg}!important;}}
[data-testid="stSidebarCollapsedControl"] *,[data-testid="collapsedControl"] *,[data-testid="stSidebarCollapseButton"] *,.stApp button svg,.stApp button [data-testid="stIconMaterial"]{{color:{fg}!important;fill:{fg}!important;}}
hr{{border-color:{border}!important;}}
.st-key-greedy-step-control [data-testid="stSliderTickBarMin"],.st-key-greedy-step-control [data-testid="stSliderTickBarMax"]{{visibility:hidden;}}
.greedy-ticks{{display:flex;justify-content:space-between;padding:0 .1rem;margin-top:-.6rem;margin-bottom:1.2rem;color:{fg};}}
.greedy-ticks span{{display:flex;flex-direction:column;align-items:center;gap:.25rem;font-size:.9rem;width:0;min-width:0;}}
.greedy-ticks i{{display:block;width:2px;height:10px;background:{muted};}}
.greedy-ticks span[data-active="true"]{{color:#d88b68;font-weight:700;}}
.greedy-ticks span[data-active="true"] i{{background:#d88b68;height:12px;}}
.table-scroll{{max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid {border};border-radius:4px;margin:.7rem 0 1.3rem;}}
.research-table{{width:100%;border-collapse:collapse!important;background:{bg}!important;color:{fg}!important;font-size:.88rem;}}
.research-table th,.research-table td{{padding:.65rem .75rem;text-align:left!important;border:0!important;border-bottom:1px solid {border}!important;color:{fg}!important;white-space:normal;min-width:5rem;}}
.research-table th{{background:{surface}!important;font-weight:600;}}
.research-table tr:last-child td{{border-bottom:0!important;}}
.research-table tbody tr:nth-child(even){{background:{surface}!important;}}
.research-table th:first-child,.research-table td:first-child{{min-width:2rem;}}
[data-testid="stLatex"]{{overflow:visible!important;padding-top:1.1rem!important;padding-bottom:.8rem!important;}}
.katex-display{{overflow-x:auto!important;overflow-y:hidden!important;padding:.8rem .25rem!important;margin:.4rem 0!important;}}
.katex{{line-height:1.5!important;}}
@media(max-width:760px){{
.block-container{{padding:3.3rem .9rem 2rem!important;}}
h1{{font-size:2.7rem!important;line-height:1.08!important;}}
h2{{font-size:2rem!important;}}
[data-testid="stHorizontalBlock"]{{flex-direction:column!important;gap:1rem!important;}}
[data-testid="stColumn"]{{width:100%!important;min-width:0!important;flex:1 1 100%!important;}}
[data-testid="stMetricValue"]{{font-size:1.8rem!important;}}
.stButton button,.stDownloadButton button{{min-height:44px;}}
.katex{{font-size:1rem!important;}}
.research-table{{font-size:.8rem;}}
.research-table th,.research-table td{{padding:.55rem;}}
}}
</style>'''
