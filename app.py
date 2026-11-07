"""Run: python -m streamlit run app.py"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import html,io,json,logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from appcore.constants import *
from appcore.data import results,rows,sequence_rows,manifest
from appcore.charts import style,body,COLORS
from appcore.kinematics import fk,generate
from appcore.text import COPY
from appcore.smpl_body import placement
from appcore.playback import playback
from appcore.ui import numbered_table,theme_css
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
st.set_page_config(page_title='Sparse IMU · Placement matters',page_icon='◉',layout='wide',initial_sidebar_state='auto')
with st.sidebar:
    st.markdown('### ◉  SPARSE / IMU')
    lang=st.selectbox('Language / Bahasa',['en','id'],format_func=lambda x:'English' if x=='en' else 'Bahasa Indonesia')
    def t(en,id):return en if lang=='en' else id
    dark=st.toggle(t('Dark mode','Mode gelap'),value=False,key='dark_mode')
    pages=[t('Overview','Ringkasan'),t('Research','Penelitian'),t('Search explorer','Jelajah pencarian'),t('Recorded results','Hasil eksperimen'),t('Motion lab','Laboratorium gerak'),t('Mapping & access','Pemetaan & akses')]
    st.session_state.setdefault('page_index',0)
    for i,title in enumerate(pages):
        active=st.session_state.page_index==i
        if st.button(title,key=f'nav_{i}',type='primary' if active else 'tertiary',width='stretch'):
            st.session_state.page_index=i
            st.rerun()
    page=pages[st.session_state.page_index]
    st.caption(t('13 candidate sites · 6 selected\nPelvis always included','13 lokasi kandidat · 6 dipilih\nPanggul selalu disertakan'))
    st.divider();st.caption(t('Research companion · trained on CMU AMASS · evaluated on DIP-IMU and TotalCapture','Pendamping riset · dilatih pada CMU AMASS · dievaluasi pada DIP-IMU dan TotalCapture'))
fg,bg,muted,border=('#eee9e0','#181d1a','#bbc2b7','#3c453c') if dark else ('#292d28','#faf8f2','#657064','#dadbd0')
st.markdown(f'''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Libre+Caslon+Display&display=swap');
html,body,[class*="css"],.stApp,p,div,button,input,select{{font-family:'DM Sans',sans-serif;}}
.stApp,[data-testid="stHeader"]{{background:{bg};color:{fg};}}
[data-testid="stSidebar"]{{background:{'#202720' if dark else '#efeee5'};color:{fg};border-right:1px solid {border};}}
.block-container{{padding-top:2.5rem;padding-bottom:4rem;max-width:1450px;}}
h1,h2,h3{{color:{fg};letter-spacing:-.035em;}}
h1{{font-family:'Libre Caslon Display',Georgia,serif!important;font-size:clamp(3rem,5vw,5.8rem)!important;line-height:1.04!important;font-weight:400!important;}}
h2{{font-family:'Libre Caslon Display',Georgia,serif!important;font-size:2.7rem!important;font-weight:400!important;}}
h3{{font-size:1.2rem!important;font-weight:600!important;}}
p,li{{line-height:1.7;}}
[data-testid="stCaptionContainer"]{{color:{muted};}}
[data-testid="stMetric"]{{border-top:1px solid {border};padding-top:1rem;}}
[data-testid="stMetricValue"]{{font-size:2rem;}}
.stButton button{{border-radius:2px;border:1px solid {border};}}
.stButton button[kind="primary"]{{background:#ad5437;color:#fff;border-color:#ad5437;}}
hr{{border-color:{border};}}
.eyebrow{{color:{muted};font-size:.75rem;letter-spacing:.18em;text-transform:uppercase;margin-bottom:1rem;}}
.lead{{font-size:1.15rem;max-width:780px;color:{muted};line-height:1.7;}}
[data-baseweb="select"]>div,[data-baseweb="input"],textarea{{background:{'#273027' if dark else '#fffdf7'}!important;color:{fg}!important;}}
[data-testid="stMarkdownContainer"]{{color:{fg};}}
</style>''',unsafe_allow_html=True)
st.markdown(theme_css(dark),unsafe_allow_html=True)
def copy(key):return COPY[key][lang=='id']
def eyebrow(s):st.markdown(f'<div class="eyebrow">{html.escape(s)}</div>',unsafe_allow_html=True)
def chart(f):st.plotly_chart(f,use_container_width=True,theme=None,config={'displaylogo':False,'responsive':True})
def label(s):
    if s=='upper_13':return t('All 13','Semua 13')
    return LABELS.get(s,s.replace('random_','Random '))
def metric_name(k):return METRICS[k][lang=='id']
def dataset_name(k):return t('CMU validation','Validasi CMU') if k=='synthetic_validation' else DATASETS[k]
def choose(key,all_labels=False):
    values=list(LABELS)
    return st.selectbox(t('Configuration','Konfigurasi'),values,format_func=label,key=key)
def metric_cards(row):
    cols=st.columns(3)
    for i,(k,(_,__,unit)) in enumerate(METRICS.items()):
        v=row.get(k)
        cols[i%3].metric(metric_name(k),'N/A' if v is None or pd.isna(v) else f'{v:.3f} {unit}')
d=results();df=rows();names=NAMES if lang=='en' else NAMES_ID
if page==pages[0]:
    eyebrow(t('A study of sparse inertial pose estimation','Studi estimasi pose inersial dengan sensor terbatas'))
    left,right=st.columns([1.2,1],gap='large')
    with left:
        st.title(copy('hero'));st.markdown(f'<p class="lead">{copy("deck")}</p>',unsafe_allow_html=True)
        st.caption(t('BINUS University · Computer Science & Mathematics','Universitas BINUS · Ilmu Komputer & Matematika'))
    with right:
        selected=choose('overview_layout');joints=manifest()[ALIASES.get(selected,selected)]['joints'];_,pos=fk(np.broadcast_to(np.eye(3),(24,3,3)))
        view=st.selectbox(t('Viewing angle','Sudut pandang'),['three_quarter','front','side'],format_func=lambda x:{'three_quarter':t('Three-quarter','Tiga perempat'),'front':t('Front','Depan'),'side':t('Side','Samping')}[x],key='overview_view');chart(placement(joints,dark,lang,view=view));st.caption(t('Public fixed-shape SMPL-Body · illustrative anatomical markers, not calibrated mounts. Drag to rotate. Scroll to zoom.','SMPL-Body publik berbentuk tetap · penanda anatomis ilustratif, bukan pemasangan terkalibrasi. Seret untuk memutar. Gulir untuk zoom.'));st.caption('SMPL-Body © Max Planck Institute for Intelligent Systems · Meshcapade sample · [CC BY 4.0](https://smpl.is.tue.mpg.de/bodylicense.html)')
    st.divider();a,b=st.columns([1,1.5],gap='large')
    with a:st.header(copy('question'))
    with b:st.write(copy('question_body'));st.write(' · '.join(f'{j}: {names[j]}' for j in joints))
    st.write(copy('contribution'))
    st.divider();st.header(copy('finding'));st.write(copy('finding_body'))
    c=st.columns(3)
    c[0].metric(t('Greedy A · CMU local error','Greedy A · galat lokal CMU'),'8.089°')
    c[1].metric(t('Greedy B · DIP-IMU MPJPE','Greedy B · MPJPE DIP-IMU'),'7.743 cm')
    c[2].metric(t('Admissible six-site layouts','Konfigurasi enam lokasi yang valid'),'792')
    st.caption(t('Five-epoch recorded results. Different metrics and datasets. These values are not directly comparable.','Hasil tercatat lima epoch. Metrik dan dataset berbeda. Nilai ini tidak dibandingkan langsung.'))
elif page==pages[1]:
    eyebrow(t('01 / Research','01 / Penelitian'));st.title(t('What was held fixed.\nWhat was searched.','Apa yang dipertahankan.\nApa yang dicari.'))
    st.header(copy('background'));st.write(copy('background_body'))
    for heading,body_key in [('protocol','protocol_body'),('synthesis','synthesis_body'),('model','model_body')]:
        st.header(copy(heading));st.write(copy(body_key))
        if heading=='protocol':
            st.write(t('CMU AMASS motion is synthesized into IMU signals and split by capture group. Training fits each candidate model. Validation guides random/greedy selection and swap refinement. Frozen layouts are retrained for five epochs, then evaluated on DIP-IMU and TotalCapture.','Gerakan CMU AMASS disintesis menjadi sinyal IMU dan dibagi berdasarkan kelompok perekaman. Data latih digunakan untuk model kandidat. Validasi memandu pemilihan acak/greedy dan refinement swap. Konfigurasi tetap dilatih ulang lima epoch, lalu dievaluasi pada DIP-IMU dan TotalCapture.'))
        if heading=='synthesis':
            st.latex(r'R^g_{t,0}=R^l_{t,0},\qquad R^g_{t,j}=R^g_{t,p(j)}R^l_{t,j}')
            st.caption(t('Global orientations compose parent and local rotations. p(j) is the parent of joint j.','Orientasi global menggabungkan rotasi induk dan lokal. p(j) adalah induk sendi j.'))
            st.latex(r'O_{t,j}=R^g_{t,j}')
            st.latex(r'a_{t,j}=\frac{3600}{16}(v_{t+4,j}-2v_{t,j}+v_{t-4,j})')
            st.latex(r'a_{t,j}=3600(v_{t+1,j}-2v_{t,j}+v_{t-1,j})')
            st.caption(t('Four-frame differences in the interior. Adjacent-frame differences near the boundaries. Endpoints are zero. Positions v are in metres.','Beda empat frame di bagian tengah. Beda satu frame di dekat batas. Ujung bernilai nol. Posisi v dalam meter.'))
            st.write(copy('normalization'))
            st.latex(r'\widetilde a_{t,0}=(O_{t,0}^{T}a_{t,0})/30')
            st.latex(r'\widetilde a_{t,j}=O_{t,0}^{T}(a_{t,j}-a_{t,0})/30,\quad j\ne0')
            st.latex(r'\widetilde O_{t,0}=O_{t,0},\qquad\widetilde O_{t,j}=O_{t,0}^{T}O_{t,j}')
            st.caption(t('Acceleration vectors are columns. T denotes transpose. The fixed divisor 30 scales acceleration relative to orientation channels.','Vektor percepatan berbentuk kolom. T menyatakan transpose. Pembagi tetap 30 menskalakan percepatan relatif terhadap kanal orientasi.'))
    st.header(t('External evaluation','Evaluasi eksternal'));st.write(copy('evaluation_body'))
    st.header(t('Six views of reconstruction quality','Enam ukuran kualitas rekonstruksi'))
    definitions=[t('Mean geodesic error over 24 parent-relative rotations. Root is world-relative.','Rata-rata galat geodesik 24 rotasi relatif induk. Akar relatif dunia.'),t('Mean geodesic error after composing all 24 rotations through the hierarchy.','Rata-rata galat geodesik setelah menyusun 24 rotasi melalui hierarki.'),t('Global rotation error at both thighs and upper arms: indices 1, 2, 16, 17.','Galat rotasi global kedua paha dan lengan atas: indeks 1, 2, 16, 17.'),t('Mean distance over 24 mean-shape joints, pelvis translation aligned.','Rata-rata jarak 24 sendi bentuk rata-rata, disejajarkan pada translasi panggul.'),t('Mean corresponding-vertex distance, pelvis translation aligned.','Rata-rata jarak vertex bersesuaian, disejajarkan pada translasi panggul.'),t('Magnitude of third temporal position differences at 60 Hz. Smoothness, not accuracy.','Besar beda temporal orde tiga posisi pada 60 Hz. Kehalusan, bukan akurasi.')]
    numbered_table(pd.DataFrame({t('Metric','Metrik'):[metric_name(k) for k in METRICS],t('Unit','Satuan'):[v[2] for v in METRICS.values()],t('Definition','Definisi'):definitions}))
    st.info(t('Search uses only frame-weighted local rotation error. Final recorded tables use equal-sequence means. MPJPE and the other metrics do not select the layouts.','Pencarian hanya menggunakan galat rotasi lokal berbobot frame. Tabel akhir menggunakan rata-rata dengan bobot sama antarsekuens. MPJPE dan metrik lain tidak digunakan untuk memilih konfigurasi.'))
    st.header(t('How a layout is scored','Cara menilai konfigurasi'))
    st.write(t('The search score is the mean local geodesic rotation error over all valid validation frames and 24 joints. Final tables instead give each sequence equal weight.','Skor pencarian adalah rata-rata galat geodesik rotasi lokal seluruh frame validasi valid dan 24 sendi. Tabel akhir memberi bobot sama pada setiap sekuens.'))
    st.latex(r'D_{t,j}=\widehat R_{t,j}^{T}R_{t,j}')
    st.latex(r'\theta_{t,j}=\frac{180}{\pi}\arccos\left(\frac{\operatorname{tr}(D_{t,j})-1}{2}\right)')
    st.latex(r'E(S)=\frac{1}{24F}\sum_{t=1}^{F}\sum_{j=0}^{23}\theta_{t,j}(S)')
    st.caption(t('S is the selected layout and F the valid-frame count. The arccos input is clipped to [-1,1].','S adalah konfigurasi terpilih dan F jumlah frame valid. Masukan arccos dibatasi ke [-1,1].'))
    st.header(t('Greedy B: error band and MIC','Greedy B: rentang galat dan MIC'))
    st.latex(r'e_j\leq e_{\min}+\tau,\qquad\tau=0.25^\circ')
    st.latex(r'M_{i,j}=\frac{1}{144}\sum_{a=1}^{12}\sum_{b=1}^{12}\operatorname{MIC}(x_{i,a},x_{j,b})')
    st.latex(r'r_j=\operatorname{mean}_{u\in U}M_{j,u},\qquad d_j=\operatorname{mean}_{s\in S}M_{j,s}')
    st.latex(r'q_j=r_j-d_j')
    st.write(t('U contains unselected sensors except candidate j. S contains selected sensors including pelvis. Greedy B takes the highest q within the error band. This measures sensor-to-sensor dependence, not dependence on the pose target.','U berisi sensor belum terpilih selain kandidat j. S berisi sensor terpilih termasuk panggul. Greedy B memilih q tertinggi dalam rentang galat. Ini mengukur ketergantungan antarsensor, bukan terhadap target pose.'))
    st.header(copy('limits'));st.write(copy('limits_body'))
elif page==pages[2]:
    eyebrow(t('02 / Search','02 / Pencarian'));st.title(t('Follow the decisions.','Ikuti proses pemilihannya.'))
    st.write(copy('search_intro'));st.write(copy('random'));st.write(copy('greedy'));st.write(copy('mic'))
    st.latex(r'\mathrm{eligible}:\quad e_j\leq e_{\min}+0.25^\circ');st.latex(r'\mathrm{score}(j)=\mathrm{mean\ MIC\ with\ unselected}-\mathrm{mean\ MIC\ with\ selected}')
    fig=go.Figure()
    for key in ['greedy_a','greedy_b_tau_0p25']:
        steps=d[key]['steps'];fig.add_trace(go.Scatter(x=[s['k'] for s in steps],y=[s['chosen_error_deg'] for s in steps],mode='lines+markers',name=label(key)))
    fig.update_layout(xaxis_title=t('Selected sites, including pelvis','Lokasi terpilih, termasuk panggul'),yaxis_title=t('3-epoch validation local error (°)','Galat lokal validasi 3 epoch (°)'))
    chart(style(fig,dark))
    variant=st.radio(t('Inspect a path','Periksa jalur'),['greedy_a','greedy_b_tau_0p25'],format_func=label,horizontal=True)
    g=d[variant]
    with st.container(key='greedy-step-control'):
        step=st.slider(t('Greedy step','Langkah greedy'),1,5,1,step=1,key='greedy_step')
        ticks=''.join(f'<span data-active="{str(i==step).lower()}"><i></i>{i}</span>' for i in range(1,6))
        st.markdown('<div class="greedy-ticks" aria-hidden="true">'+ticks+'</div>',unsafe_allow_html=True)
    s=g['steps'][step-1]
    a,b=st.columns([1.3,1])
    with a:
        candidates=pd.DataFrame(s['candidates']);candidates['site']=[f'{j}: {names[j]}' for j in candidates.added_joint]
        f=go.Figure(go.Bar(x=candidates['local_rotation_deg'],y=candidates['site'],orientation='h',marker_color=['#bf6849' if j==s['chosen_joint'] else '#aab8ac' for j in candidates.added_joint]));f.update_layout(xaxis_title=t('Local error (°)','Galat lokal (°)'),yaxis=dict(autorange='reversed'));chart(style(f,dark,460))
    with b:
        st.subheader(t('Chosen addition','Tambahan terpilih'));st.write(f"{s['chosen_joint']}: {names[s['chosen_joint']]}");st.write(t('Selected so far: ','Terpilih sejauh ini: ')+', '.join(names[j] for j in s['selected']))
        st.metric(t('Chosen error','Galat terpilih'),f"{s['chosen_error_deg']:.4f}°")
        if variant!='greedy_a':
            st.write(t('Error-band candidates: ','Kandidat dalam rentang galat: ')+', '.join(names[j] for j in s['tie_set']));st.write(t('MIC tie-breaker triggered: ','Tie-breaker MIC terpicu: ')+str(s['tie_breaker_triggered']))
            if s['tie_scores']:numbered_table(pd.DataFrame(s['tie_scores']),hide_index=True,width='stretch')
    st.write(copy('mic_detail'))
    st.header(t('Reconsidering an earlier choice','Meninjau pilihan sebelumnya'));st.write(copy('swap'))
    f=go.Figure()
    for key in ['greedy_a_swap','greedy_b_tau_0p25_swap']:
        g=d[key];errors=[g['initial']['local_rotation_deg']]+[r['best_neighbor']['local_rotation_deg'] if r['accepted'] else r['before']['local_rotation_deg'] for r in g['rounds']]
        f.add_trace(go.Scatter(x=list(range(len(errors))),y=errors,mode='lines+markers',name=label(key)))
    f.update_layout(xaxis_title=t('Completed sweeps (0 = initial)','Putaran selesai (0 = awal)'),yaxis_title=t('3-epoch local error (°)','Galat lokal 3 epoch (°)'));chart(style(f,dark))
    numbered_table(pd.DataFrame(d['swap_summary'])[['label','accepted_swaps','sweeps','initial_error_deg','refined_error_deg','local_optimum_certified']],hide_index=True,width='stretch')
elif page==pages[3]:
    eyebrow(t('03 / Recorded evidence','03 / Hasil tercatat'));st.title(t('Compare like with like.','Bandingkan pada kondisi yang sama.'))
    dataset=st.selectbox(t('Evaluation dataset','Dataset evaluasi'),list(DATASETS),format_func=dataset_name)
    metric=st.selectbox(t('Distribution metric','Metrik distribusi'),list(METRICS),format_func=lambda k:f'{metric_name(k)} ({METRICS[k][2]})',index=0 if dataset=='synthetic_validation' else 3)
    subset=df[(df.dataset==dataset)&(df.condition=='original')&(df.status=='complete')];random=subset[subset.family=='random']
    st.caption(t('Final budget: 5 epochs from initialization. Seed 1234. Equal-sequence means. Lower error is better. Lower jitter only indicates smoother motion.','Anggaran akhir: 5 epoch dari inisialisasi. Seed 1234. Rata-rata berbobot sama antarsekuens. Galat lebih rendah lebih baik. Jitter lebih rendah hanya menunjukkan gerakan lebih halus.'))
    if dataset=='synthetic_validation':st.warning(t('This validation partition was also used for selection. It is not an independent test.','Partisi validasi ini juga digunakan untuk pemilihan. Ini bukan pengujian independen.'))
    f=go.Figure(go.Histogram(x=random[metric],nbinsx=12,name=t('30 random layouts','30 konfigurasi acak'),marker_color='#bbcdbb'))
    for i,key in enumerate(['sip_dip','greedy_a','greedy_b_tau_0p25','upper_13']):
        row=subset[subset.label==key].iloc[0];f.add_vline(x=row[metric],line_color=COLORS[i],line_dash=['dot','solid','dash','dashdot'][i]);f.add_trace(go.Scatter(x=[None],y=[None],mode='lines',line=dict(color=COLORS[i]),name=label(key)+(t(' / both swaps',' / kedua swap') if key=='greedy_a' else '')+f" · {row[metric]:.3f}"))
    f.update_layout(xaxis_title=f'{metric_name(metric)} ({METRICS[metric][2]})',yaxis_title=t('Layouts','Konfigurasi'),bargap=.08);chart(style(f,dark,430))
    selected=choose('result_layout');row=subset[subset.label==selected].iloc[0];st.subheader(label(selected));metric_cards(row)
    pct=row.get(metric+'_random_error_percentile');st.caption(t('Empirical random error percentile: ','Persentil galat acak empiris: ')+f'{pct:.1f}%. '+t('Lower is better. Ties use half weight. A percentile of zero is not proof of global optimality.','Lebih rendah lebih baik. Ikatan berbobot setengah. Persentil nol bukan bukti optimum global.'))
    reference=subset[subset.label=='sip_dip'].iloc[0]
    ratios=[row[k]/reference[k] for k in METRICS];f=go.Figure(go.Bar(x=[metric_name(k) for k in METRICS],y=ratios,marker_color='#bf6849'));f.add_hline(y=1,line_dash='dot');f.update_layout(yaxis_title=t('Ratio to SIP/DIP (1 = equal)','Rasio terhadap SIP/DIP (1 = sama)'));chart(style(f,dark))
    st.caption(t('Ratios allow different units on one chart. They do not combine the six metrics into one score.','Rasio memungkinkan satu grafik dengan satuan berbeda. Keenam metrik tidak digabung menjadi satu skor.'))
    comparison=subset[subset.label.isin(LABELS)].copy();comparison['configuration']=comparison.label.map(label)
    comparison_table=comparison[['configuration',*METRICS]].rename(columns={k:f'{metric_name(k)} ({v[2]})' for k,v in METRICS.items()});numbered_table(comparison_table,hide_index=True,width='stretch')
    st.download_button(t('Download all recorded metrics (CSV)','Unduh seluruh metrik (CSV)'),df.to_csv(index=False).encode(),'recorded_metrics.csv','text/csv')
    st.header(t('Sequence-level inspection','Pemeriksaan per sekuens'));seq=sequence_rows();seq=seq[(seq.dataset==dataset)&(seq.label==selected)&(seq.condition=='original')].copy()
    if not seq.empty:
        seq=seq.sort_values(metric);st.caption(t('Best and worst here refer to sequence-average values of the selected metric, not frame extrema. Pose playback is unavailable without permitted recordings/predictions.','Terbaik dan terburuk di sini berdasarkan rata-rata sekuens metrik terpilih, bukan ekstrem frame. Pemutaran pose tidak tersedia tanpa rekaman/prediksi yang diizinkan.'))
        numbered_table(pd.concat([seq.head(3),seq.tail(3)]).drop_duplicates()[['id',*METRICS]],hide_index=True,width='stretch')
    st.header(t('Discussion from the paper','Pembahasan dari paper'));st.write(copy('results_'+dataset));st.write(copy('refinement_result'))
    if dataset in ('dip','totalcapture'):
        st.subheader(t('Pose examples discussed in the paper','Contoh pose yang dibahas dalam paper'));st.write(copy('qualitative_'+dataset))
elif page==pages[4]:
    eyebrow(t('04 / Live inference','04 / Inferensi langsung'));st.title(t('See what the network does.','Lihat cara model bekerja.'));st.warning(copy('demo_warning'))
    a,b=st.columns([1,1.5],gap='large')
    with a:
        selected=choose('lab_layout');motion=st.selectbox(t('Generated motion','Gerakan buatan'),['walk','reach','bend'],format_func=lambda x:{'walk':t('Walking pattern','Pola berjalan'),'reach':t('Arm reach','Jangkauan lengan'),'bend':t('Forward bend','Membungkuk')}[x]);speed=st.slider(t('Cycles per second','Siklus per detik'),.25,1.5,.75,.05);amplitude=st.slider(t('Motion amplitude','Amplitudo gerakan'),.25,1.5,1.,.05)
        run=st.button(t('Run trained model','Jalankan model terlatih'),type='primary',width='stretch')
        st.caption(t('4 seconds · 240 frames · CPU FP32 · no training','4 detik · 240 frame · CPU FP32 · tanpa pelatihan'))
        st.info(t('Real DIP-IMU / TotalCapture results are available under Recorded results. This public lab cannot rerun those recordings.','Hasil DIP-IMU / TotalCapture tersedia pada Hasil eksperimen. Laboratorium publik ini tidak dapat menjalankan ulang rekaman tersebut.'))
    signature=(selected,motion,speed,amplitude)
    if run:
        from appcore.inference import predict,demo_metrics
        import time
        with st.spinner(t('Running checkpoint and computing demonstration metrics…','Menjalankan checkpoint dan menghitung metrik demonstrasi…')):
            generated=generate(motion,speed=speed,amplitude=amplitude);start=time.perf_counter();pred=predict(selected,generated['acc'],generated['ori']);metrics=demo_metrics(pred,generated['pose']);st.session_state['demo']={'signature':signature,'metrics':metrics,'seconds':time.perf_counter()-start}
    saved=st.session_state.get('demo')
    with b:
        if saved and saved['signature']==signature:
            m=saved['metrics'];playback(m['ref_positions'],m['pred_positions'],manifest()[ALIASES.get(selected,selected)]['joints'],m['per_frame_mpjpe'],dark,lang)
            st.caption(t('Playback shows the generated schematic body. The static placement viewer uses the public SMPL-Body.','Pemutaran menggunakan tubuh skematis buatan. Penampil lokasi statis menggunakan SMPL-Body publik.'))
        else:
            _,pos=fk(np.broadcast_to(np.eye(3),(24,3,3)));chart(body(pos,manifest()[ALIASES.get(selected,selected)]['joints'],dark,lang));st.caption(t('Choose a motion and run the model.','Pilih gerakan lalu jalankan model.'))
    if saved and saved['signature']==signature:
        st.subheader(t('Generated-motion demonstration only','Hanya demonstrasi gerakan buatan'));metric_cards(saved['metrics']);st.caption(t('Mesh error is unavailable for schematic-motion inference. The static SMPL-Body is for placement visualization only.','Galat mesh tidak tersedia untuk inferensi gerakan skematis. SMPL-Body statis hanya untuk visualisasi lokasi.'))
        f=go.Figure(go.Scatter(x=np.arange(240)/60,y=saved['metrics']['per_frame_mpjpe'],line=dict(color='#bf6849'),name='MPJPE'));f.update_layout(xaxis_title=t('Time (s)','Waktu (s)'),yaxis_title=t('Schematic MPJPE (cm)','MPJPE skematis (cm)'));chart(style(f,dark,300))
        st.caption(t('Processing time: ','Waktu pemrosesan: ')+f"{saved['seconds']:.2f} s")
elif page==pages[5]:
    eyebrow(t('05 / Correspondence & access','05 / Korespondensi & akses'));st.title(t('Thirteen sites.\nTwo real sensor systems.','Tiga belas lokasi.\nDua sistem sensor nyata.'));st.write(copy('access'))
    st.warning(t('Mappings describe body segments, not identical physical mounting coordinates. The DIP upper-torso stream is an asymmetric left-chest proxy. Approval does not make it exact.','Pemetaan menggambarkan segmen tubuh, bukan koordinat pemasangan fisik yang identik. Sinyal batang tubuh atas DIP menggunakan proksi dada kiri asimetris. Persetujuan tidak menjadikannya persis.'))
    mapping=pd.DataFrame(d['mapping']['rows']);mapping['segment']=[names[j] for j in mapping.smpl_index]
    numbered_table(mapping[['smpl_index','segment','dip_index','dip_name','totalcapture_index','totalcapture_name','dip_confidence','totalcapture_confidence']],hide_index=True,width='stretch')
    st.header(t('Data sources','Sumber data'))
    st.markdown('[AMASS](https://amass.is.tue.mpg.de/) · [DIP](https://dip.is.tuebingen.mpg.de/) · [TotalCapture](https://cvssp.org/data/totalcapture/) · [SMPL](https://smpl.is.tue.mpg.de/)')
    st.write(t('Registration and provider-specific terms apply. Download data through the providers. Do not put restricted assets in a public GitHub repository.','Registrasi dan ketentuan masing-masing penyedia berlaku. Unduh melalui penyedia. Jangan masukkan aset terbatas ke repositori GitHub publik.'))
    st.markdown(t('[Research code](https://github.com/FlxdeCat/optimizing-sensor-placement-for-sparse-imu-human-pose-estimation-via-combinatorial-search)', '[Kode penelitian](https://github.com/FlxdeCat/optimizing-sensor-placement-for-sparse-imu-human-pose-estimation-via-combinatorial-search)'))
    st.header(t('Reproducibility & provenance','Reproduksibilitas & asal data'))
    st.write(t('The site uses the supplied completed analysis and five final checkpoints. The exported weight-only bundles preserve model tensors and checkpoint identity while removing optimizer and RNG state. Checksums are verified before inference. No upstream source code or raw recording is included.','Situs menggunakan analisis selesai dan lima checkpoint akhir yang diberikan. Paket bobot mempertahankan tensor model dan identitas checkpoint, serta menghapus status optimizer dan RNG. Checksum diverifikasi sebelum inferensi. Tidak ada kode upstream atau rekaman mentah yang disertakan.'))
    st.download_button(t('Download model provenance','Unduh asal model'),(ROOT/'assets/models.json').read_bytes(),'models.json','application/json')
st.divider();st.caption(t('Optimizing Sensor Placement for Sparse-IMU Human Pose Estimation via Combinatorial Search · Research companion · Paper link forthcoming','Optimizing Sensor Placement for Sparse-IMU Human Pose Estimation via Combinatorial Search · Pendamping penelitian · Tautan paper menyusul'))
