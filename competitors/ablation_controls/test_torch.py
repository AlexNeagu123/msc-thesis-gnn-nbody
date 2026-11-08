"""Synthetic-only tests; no TRAIN/VAL/test access or production training."""
import copy
import random
import numpy as np
import pytest
import torch
from models.hgnn_canonical import CanonicalHGNN
from .hgnn import KineticAblationHGNN,objective,acceleration_target
from .schedule import matched_tail_order,UnitCursor
from .recipes import CELLS,validate_cell
from . import torch_checkpoint as C
torch.set_num_threads(1)

def state(batch=4,dtype=torch.float64):
    s=torch.tensor([[[-.6,.2,.1,-.2,1.],[.5,.4,-.1,.2,2.],[.1,-.7,.2,.1,1.5]]],dtype=dtype)
    return s.repeat(batch,1,1)
def model(kinetic='exact',full=False):
    return KineticAblationHGNN(kinetic=kinetic,hidden_dim=64 if full else 8,n_layers=4 if full else 1,pos_std=.72,momentum_std=1.4).double()
def bank():return {'horizon':np.repeat([1,5,20,50,100],500)}

def test_all_registered_cells_reject_unregistered_combination():
    from dataclasses import replace
    assert len(CELLS)==13
    for cell in CELLS.values():assert validate_cell(cell)==cell and cell.record()['production_authorized'] is False
    with pytest.raises(ValueError):validate_cell(replace(CELLS['gnode_native'],clip_norm=1.))

def test_potential_initialization_exact_and_learned_parity_full_architecture():
    torch.manual_seed(42);a=CanonicalHGNN(hidden_dim=64,n_layers=4,pos_std=.72,momentum_std=1.4).double()
    torch.manual_seed(42);b=model('learned',True)
    torch.manual_seed(42);c=model('exact',True)
    for name,v in a.potential.state_dict().items():
        assert torch.equal(v,b.potential.state_dict()[name]);assert torch.equal(v,c.potential.state_dict()[name])
    torch.testing.assert_close(a(state()),b(state()),rtol=0,atol=0)
    assert sum(p.numel() for p in a.parameters())==191040
    assert sum(p.numel() for p in c.parameters())<191040

def test_exact_kinetic_chain_rule_energy_and_rotation():
    m=model();s=state();mass=s[...,4:];p=(mass*s[...,2:4]).requires_grad_(True)
    torch.testing.assert_close(m._velocity(p,mass,create_graph=True),s[...,2:4],rtol=1e-12,atol=1e-12)
    energy=m.kinetic(p/m.momentum_std,mass)
    torch.testing.assert_close(energy,.5*(p.square()/mass).sum((-1,-2)))
    r=torch.tensor([[0.,-1.],[1.,0.]],dtype=p.dtype);pr=(p@r.T)
    torch.testing.assert_close(m._velocity(pr,mass,create_graph=False),s[...,2:4]@r.T)

def test_mass_and_input_preservation_and_symmetries():
    m=model();s=state();before=s.clone();out=m(s)
    torch.testing.assert_close(s,before);torch.testing.assert_close(out[...,4:],s[...,4:])
    perm=[2,0,1];torch.testing.assert_close(m(s[:,perm]),out[:,perm],rtol=1e-10,atol=1e-10)
    shifted=s.clone();shifted[...,:2]+=torch.tensor([.3,-.2])
    expected=out.clone();expected[...,:2]+=torch.tensor([.3,-.2]);torch.testing.assert_close(m(shifted),expected,rtol=1e-10,atol=1e-10)

def test_exact_vs_learned_identity_rejected():
    a=model('learned');b=model('exact')
    with pytest.raises(RuntimeError):b.load_state_dict(a.state_dict(),strict=False)

def test_physics_supervises_potential_not_kinetic_and_force_mass_sign():
    s=state();m=model('learned');target=acceleration_target(s)
    torch.testing.assert_close((s[...,4:]*target).sum(1),torch.zeros_like(target[:,0]),atol=1e-12,rtol=0)
    assert target[0,0,0]>0
    targets=s[:,None].repeat(1,2,1,1)
    a=objective(m,s,targets,1.1173796653747559,'scaled')
    gs=torch.autograd.grad(a['physics'],list(m.kinetic.parameters()),allow_unused=True,retain_graph=True)
    assert all(g is None for g in gs)
    gp=torch.autograd.grad(a['physics'],list(m.potential.parameters()),allow_unused=True)
    assert any(g is not None and g.abs().sum()>0 for g in gp)

@pytest.mark.parametrize('horizon',[1,5,20,50,100])
def test_full_bptt_finite_horizons(horizon):
    m=model();s=state(2);targets=s[:,None].repeat(1,horizon,1,1)
    loss=objective(m,s,targets,1.1173796653747559,'scaled')['total'];loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

def test_late_target_changes_parameter_gradient():
    m=model();s=state(2);y=s[:,None].repeat(1,5,1,1)
    a=torch.autograd.grad(objective(m,s,y,1.117,'off')['total'],list(m.parameters()))
    y[:,-1,:,:2]+=.01
    b=torch.autograd.grad(objective(m,s,y,1.117,'off')['total'],list(m.parameters()))
    assert any(not torch.equal(x,z) for x,z in zip(a,b))

def test_microbatch_full_gradient_parity():
    m=model();s=state(4);s[2:,:,0]+=.04;y=s[:,None].repeat(1,5,1,1)
    full=torch.autograd.grad(objective(m,s,y,1.117,'scaled')['total'],list(m.parameters()))
    acc=[torch.zeros_like(g) for g in full]
    for start in [0,2]:
        gs=torch.autograd.grad(objective(m,s[start:start+2],y[start:start+2],1.117,'scaled')['total'],list(m.parameters()))
        acc=[a+.5*g for a,g in zip(acc,gs)]
    for a,b in zip(acc,full):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)

def test_matched_tail_all_units_and_restore():
    order=matched_tail_order(bank());np.testing.assert_array_equal(order[-500:],np.arange(2000,2500))
    np.testing.assert_array_equal(np.sort(order),np.arange(2500));assert not np.array_equal(order[:2000],np.arange(2000))
    c=UnitCursor(order)
    for _ in range(2001):c.next_unit()
    saved=c.state();d=UnitCursor(order);d.restore(saved);assert c.next_unit()==d.next_unit()==2001
    with pytest.raises(ValueError):UnitCursor(np.arange(2500)).restore(saved)
    with pytest.raises(ValueError):matched_tail_order({'horizon':np.ones(2500)})

def test_exact_resume_model_adam_cursor_rng_and_progress(tmp_path):
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    a=model();opt=torch.optim.AdamW(a.parameters(),lr=2.5e-4,weight_decay=1e-5);cursor=UnitCursor(matched_tail_order(bank()));s=state(2)
    def update(m,o,c,h):
        c.next_unit();o.zero_grad();loss=objective(m,s,s[:,None].repeat(1,h,1,1),1.117,'scaled')['total'];loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
    update(a,opt,cursor,1);identity={'cell':'hgnn_exact_gravity_scaled','seed':42,'unit_order':cursor.order_hash};path=tmp_path/'state.pt'
    C.save(path,a,opt,cursor,identity,{'updates':1,'clip_history':[1.]})
    draws=(torch.rand(2),np.random.random(2),random.random());update(a,opt,cursor,5)
    b=model();other=torch.optim.AdamW(b.parameters(),lr=2.5e-4,weight_decay=1e-5);d=UnitCursor(cursor.order)
    progress=C.restore(path,b,other,d,identity);assert progress['updates']==1
    assert torch.equal(draws[0],torch.rand(2));np.testing.assert_array_equal(draws[1],np.random.random(2));assert draws[2]==random.random()
    update(b,other,d,5)
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)
    for x,y in zip(opt.state.values(),other.state.values()):
        for key in x:assert torch.equal(x[key],y[key])
    assert cursor.state()==d.state()
    with pytest.raises(ValueError):C.restore(path,b,other,d,{**identity,'seed':43})
    with pytest.raises(FileExistsError):C.save(path,a,opt,cursor,identity,{})
