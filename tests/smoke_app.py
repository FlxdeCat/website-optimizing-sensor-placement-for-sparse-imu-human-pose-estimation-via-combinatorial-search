"""Exercise all pages in EN/light and ID/dark, plus live inference."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
app=Path(__file__).resolve().parents[1]/'app.py'
for language in ['en','id']:
    at=AppTest.from_file(str(app),default_timeout=90).run()
    at.sidebar.selectbox[0].set_value(language).run()
    if language=='id':at.sidebar.toggle[0].set_value(True).run()
    for i in range(6):
        at.sidebar.button(key=f'nav_{i}').click().run()
        assert not at.exception,[(e.message,e.stack_trace) for e in at.exception]
        if i==2:
            at.main.radio[0].set_value('greedy_b_tau_0p25').run()
            assert not at.exception,[e.message for e in at.exception]
        if i==3:
            for dataset in ['dip','totalcapture','synthetic_validation']:
                at.main.selectbox[0].set_value(dataset).run()
                assert not at.exception,[e.message for e in at.exception]
        if i==4:
            at.main.button[0].click().run()
            assert not at.exception,[e.message for e in at.exception]
    print(language,'navigation, dataset controls and live inference passed')
# Check rendered plain-language text on every page in both languages.
from html.parser import HTMLParser
class Visible(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.parts=[]
    def handle_data(self,text):self.parts.append(text)
for language in ['en','id']:
    at=AppTest.from_file(str(app),default_timeout=90).run()
    at.sidebar.selectbox[0].set_value(language).run()
    for i in range(6):
        at.sidebar.button(key=f'nav_{i}').click().run()
        assert not at.exception,[e.message for e in at.exception]
        for kind in ['markdown','caption','header','subheader','title','info','warning']:
            for element in getattr(at,kind):
                value=element.value
                if '<style>' in value:continue
                parser=Visible();parser.feed(value);text=''.join(parser.parts)
                assert ';' not in text and '\u2014' not in text,(language,i,kind,text)
    print(language,'visible narrative punctuation passed')
