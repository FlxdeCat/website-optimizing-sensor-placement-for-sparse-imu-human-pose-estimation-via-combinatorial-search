import sys,unittest,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from appcore.smpl_body import asset,placement
from appcore.playback import playback_html
from appcore.kinematics import generate
from appcore.ui import theme_css
class ViewerTests(unittest.TestCase):
    def test_smpl_asset(self):
        a=asset();self.assertEqual(a['vertices'].shape,(27578,3));self.assertEqual(a['markers'].shape,(13,3));self.assertGreaterEqual(a['faces'].min(),0);self.assertLess(a['faces'].max(),len(a['vertices']));self.assertTrue(np.isfinite(a['vertices']).all())
        f=placement([0,4,5,15,18,19]);self.assertEqual(f.data[0].type,'mesh3d')
        scene=f.layout.scene
        scales=[scene.aspectratio[k]/(scene[k+'axis'].range[1]-scene[k+'axis'].range[0]) for k in ['x','y','z']]
        np.testing.assert_allclose(scales,scales[0])
        self.assertEqual(placement([0],view='side').layout.scene.camera.eye.x,2.35)
    def test_playback_markup(self):
        g=generate(seconds=.2);n=len(g['positions']);h=playback_html(g['positions'],g['positions'],[0,4,5,15,18,19],np.zeros(n),True,'id')
        for name in ['play','stop','prev','next','scrub']:self.assertIn(f'id="{name}"',h)
        for value in ['Putar','Berhenti','requestAnimationFrame','Plotly.restyle']:self.assertIn(value,h)
        self.assertNotRegex(h,r'__(?:DATA|PLOTLY|PLAY|STOP|MAX)__')
    def test_theme_and_mobile(self):
        css=theme_css(True);self.assertIn('color-scheme:dark',css);self.assertIn('@media(max-width:760px)',css);self.assertIn('stMetricValue',css);self.assertIn('.research-table',css)
if __name__=='__main__':unittest.main()
