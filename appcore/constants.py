from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
POOL=(0,1,2,4,5,7,8,9,15,16,17,18,19)
PARENTS=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21)
NAMES=('Pelvis','Left thigh','Right thigh','Spine 1','Left shank','Right shank','Spine 2','Left foot','Right foot','Upper torso','Left toe','Right toe','Neck','Left collar','Right collar','Head','Left upper arm','Right upper arm','Left forearm','Right forearm','Left wrist','Right wrist','Left hand','Right hand')
NAMES_ID=('Panggul','Paha kiri','Paha kanan','Tulang belakang 1','Betis kiri','Betis kanan','Tulang belakang 2','Kaki kiri','Kaki kanan','Batang tubuh atas','Jari kaki kiri','Jari kaki kanan','Leher','Selangka kiri','Selangka kanan','Kepala','Lengan atas kiri','Lengan atas kanan','Lengan bawah kiri','Lengan bawah kanan','Pergelangan kiri','Pergelangan kanan','Tangan kiri','Tangan kanan')
LABELS={'sip_dip':'SIP/DIP','greedy_a':'Greedy A','greedy_a_swap':'Greedy A + swap','greedy_b_tau_0p25':'Greedy B','greedy_b_tau_0p25_swap':'Greedy B + swap','random_08':'Random 08','upper_13':'All 13'}
ALIASES={'greedy_a_swap':'greedy_a','greedy_b_tau_0p25_swap':'greedy_a'}
METRICS={'local_rotation_deg':('Local rotation','Rotasi lokal','°'),'global_rotation_deg':('Global rotation','Rotasi global','°'),'sip_deg':('SIP-error','Galat SIP','°'),'mpjpe_cm':('MPJPE','MPJPE','cm'),'mesh_cm':('Mesh error','Galat mesh','cm'),'jitter_km_s3':('Jitter','Jitter','km/s³')}
DATASETS={'synthetic_validation':'CMU validation','dip':'DIP-IMU','totalcapture':'TotalCapture'}
