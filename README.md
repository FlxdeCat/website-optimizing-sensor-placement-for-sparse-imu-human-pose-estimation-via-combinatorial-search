# Sparse IMU Research Website

An interactive website accompanying **Optimizing Sensor Placement for Sparse-IMU Human Pose Estimation via Combinatorial Search**. Paper link forthcoming.

Built with Streamlit, Plotly, and PyTorch. The website includes research explanations, an interactive sensor-placement viewer, search walkthroughs, recorded results, and a motion demonstration. English, Bahasa Indonesia, and light and dark modes are available.

## Run locally

Use **Python 3.11**. Open PowerShell in the folder containing `app.py` and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the address printed in the terminal, usually **http://localhost:8501**. Press **Ctrl+C** to stop the app.

A GPU and additional dataset downloads are not required. Keep the supplied `appcore/`, `assets/`, and `models/` folders alongside `app.py`.

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload the website folder's contents to a GitHub repository. Keep `app.py` and `requirements.txt` at the repository root.
2. Include `appcore/`, `assets/`, all five `models/*.pt` files, `.streamlit/config.toml`, and the attribution files. Do not upload `.venv/` or private datasets.
3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
4. Choose the repository and branch, then set the entry-point file to **`app.py`**.
5. In **Advanced settings**, select **Python 3.11**, leave secrets empty, and deploy.

Keep the supplied dependency versions. After deployment, check the viewer, language and theme switches, results pages, and Motion Lab playback.

[Official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Using the website

Choose a page from the sidebar. In the placement viewer, select a configuration and drag to rotate the body. Search Explorer shows the greedy decisions and swap results. Recorded Results lets you compare datasets and metrics.

In Motion Lab, choose a layout and motion, then select **Run trained model**. Use **Play**, **Stop**, **Previous frame**, **Next frame**, or the timeline to inspect the prediction.

Recorded Results contains the experiment's saved measurements. Motion Lab uses generated motion on a schematic skeleton and is a separate demonstration, not a new DIP-IMU or TotalCapture evaluation. The overview's fixed SMPL-Body mesh is used for placement visualization.

## Troubleshooting

- **Missing package:** install `requirements.txt` using the same Python environment that starts the app.
- **Missing model or asset:** restore the complete `models/` and `assets/` folders from the website package.
- **Blank 3D viewer:** enable WebGL or try another browser.
- **Cloud startup failure:** inspect the deployment logs and confirm that Python 3.11 and the supplied requirements are selected.

## Attribution

The public SMPL-Body sample is distributed under CC BY 4.0. See [asset attribution](assets/smpl_body/ATTRIBUTION.md) and [NOTICE.md](NOTICE.md) for credits and applicable terms.
