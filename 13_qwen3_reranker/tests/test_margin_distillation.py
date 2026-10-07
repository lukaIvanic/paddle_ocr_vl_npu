import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from margin_distillation import lr_at,margin_loss_and_score_gradient


class MarginTest(unittest.TestCase):
    def test_all_pair_gradient_and_offset_invariance(self):
        import torch
        s=torch.tensor([1.,-2.,3.,.2],requires_grad=True)
        t=torch.tensor([2.,1.,-.5,4.])
        explicit=torch.stack([(s[i]-s[j]-t[i]+t[j])**2 for i in range(4) for j in range(i+1,4)]).mean()
        grad=torch.autograd.grad(explicit,s)[0]
        loss,replay=margin_loss_and_score_gradient(s.detach(),t)
        self.assertTrue(torch.allclose(loss,explicit))
        self.assertTrue(torch.allclose(grad,replay))
        shifted,_=margin_loss_and_score_gradient(s.detach()+9,t)
        self.assertTrue(torch.allclose(loss,shifted))

    def test_microbatch_replay_matches_joint_backward(self):
        import torch
        torch.manual_seed(2)
        a=torch.nn.Linear(3,1);b=torch.nn.Linear(3,1);b.load_state_dict(a.state_dict())
        x=torch.randn(8,3);target=torch.randn(8)
        s=a(x).flatten()
        loss,_=margin_loss_and_score_gradient(s,target);loss.backward()
        with torch.no_grad(): scores=b(x).flatten()
        _,grad=margin_loss_and_score_gradient(scores,target)
        for i in range(0,8,2):
            (b(x[i:i+2]).flatten()*grad[i:i+2]).sum().backward()
        for p,q in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-6))

    def test_schedule_has_small_early_steps_and_nonzero_endpoint(self):
        self.assertAlmostEqual(lr_at(1,50,1e-6,'warmup_linear'),2e-7)
        self.assertAlmostEqual(lr_at(5,50,1e-6,'warmup_linear'),1e-6)
        self.assertAlmostEqual(lr_at(6,50,1e-6,'warmup_linear'),1e-6)
        self.assertGreater(lr_at(50,50,1e-6,'warmup_linear'),0)
        self.assertEqual(lr_at(50,50,1e-6,'constant'),1e-6)

if __name__=='__main__':unittest.main()
