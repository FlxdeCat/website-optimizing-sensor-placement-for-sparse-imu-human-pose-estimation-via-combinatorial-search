import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from appcore.data import manifest,results,rows
from appcore.inference import pack,load_model,predict,project,demo_metrics
from appcore.kinematics import generate,rotation
class Core(unittest.TestCase):
    def test_manifest_and_weights(self):
        for label in manifest():
            model=load_model(label)
            self.assertEqual(sum(p.numel() for p in model.parameters()),212248)
    def test_packing(self):
        n=4;acc=np.arange(n*72).reshape(n,24,3).astype(float);ori=np.broadcast_to(rotation(2,np.pi/2),(n,24,3,3)).copy()
        x=pack(acc,ori,[0,4]);self.assertEqual(x.shape,(n,288))
        np.testing.assert_allclose(x[:,:3],np.einsum('tji,tj->ti',ori[:,0],acc[:,0])/30,atol=1e-6)
        np.testing.assert_allclose(x[:,12:15],np.einsum('tji,tj->ti',ori[:,0],acc[:,4]-acc[:,0])/30,atol=1e-6)
        np.testing.assert_allclose(x[:,72+4*9:72+5*9],np.broadcast_to(np.eye(3).reshape(1,9),(n,9)),atol=1e-6)
        np.testing.assert_equal(x[:,3:12],0)
    def test_projection_and_metrics(self):
        g=generate();m=demo_metrics(g['pose'],g['pose'])
        for name in ['local_rotation_deg','global_rotation_deg','sip_deg','mpjpe_cm']:self.assertLess(abs(m[name]),1e-6)
        r=project(-np.eye(3));self.assertAlmostEqual(np.linalg.det(r),1)
    def test_inference_and_alias(self):
        g=generate(seconds=1.05);a=predict('greedy_a',g['acc'],g['ori']);b=predict('greedy_b_tau_0p25_swap',g['acc'],g['ori']);self.assertEqual(a.shape,(63,24,3,3));np.testing.assert_equal(a,b)
        for label in manifest():self.assertTrue(np.isfinite(predict(label,g['acc'],g['ori'])).all())
    def test_results(self):
        df=rows()
        for dataset in ['synthetic_validation','dip','totalcapture']:
            group=df[df.dataset==dataset];self.assertEqual(len(group[group.family=='random']),30)
        self.assertEqual(len(results()['mapping']['rows']),13)
if __name__=='__main__':unittest.main()
