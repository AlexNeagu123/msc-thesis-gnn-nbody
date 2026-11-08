"""Separate TF environment. Synthetic-only tests, zero production updates/data reads."""
from competitors.gnode_pignn_training import production as P
ENVIRONMENT=P.runtime()
import numpy as np
import pytest
import tensorflow as tf
from .recipes import CELLS
from .tf_adapters import GNODEEngine,PairEngine,REFERENCE
from competitors.gnode_pignn_training.engine import StepEngine
from competitors.pignn_ablation.recipes import CELLS as OLD_PI
from competitors.gnode_pignn_training.checkpoint import save_checkpoint,restore_checkpoint
from competitors.gnode_pignn_tf.models import compiled_rollout

def states(batch=4):
    a=np.array([[[-.6,.2,.1,-.2,1.],[.5,.4,-.1,.2,2.],[.1,-.7,.2,.1,1.5]]],np.float32)
    a=np.repeat(a,batch,axis=0)[...,[4,0,1,2,3]]
    a[:,0,1]+=np.arange(batch,dtype=np.float32)*.003
    return a

def targets(x,h=1):
    y=np.repeat(x[:,None,:,1:5],h,axis=1)
    for k in range(h):y[:,k,:,:2]+=x[...,3:5]*.001*(k+1)
    return y

def exact(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

def test_gnode_native_control_initial_and_three_updates_bitwise():
    a=REFERENCE.MultiEngine();b=GNODEEngine(CELLS['gnode_native']);exact(a.arrays(),b.arrays())
    x=states();y=targets(x,5)
    for _ in range(3):
        a.update(x,y,micro=2);b.update(x,y,micro=2);exact(a.arrays(),b.arrays())

def test_gnode_semiimplicit_equations_initial_weights_and_inference():
    base=REFERENCE.MultiEngine();m=GNODEEngine(CELLS['gnode_semiimplicit']);exact(base.arrays(),m.arrays())
    x=tf.constant(states());nodes=m.model.get_layer('encoder')(x);acc=m.model.get_layer('gnn')(nodes)
    v=x[...,3:5]+.001*acc;q=x[...,1:3]+.001*v
    np.testing.assert_array_equal(m.model(x).numpy(),tf.concat([q,v],-1).numpy())
    np.testing.assert_array_equal(m.model(x).numpy(),m.inference_view()(x).numpy())
    result=compiled_rollout(m.inference_view(),'gnode')(x,tf.constant(5)).numpy()
    assert result.shape==(4,5,3,5) and np.isfinite(result).all()

@pytest.mark.parametrize('horizon',[1,5,20,50,100])
def test_gnode_full_micro_gradients_finite_and_last_target(horizon):
    m=GNODEEngine(CELLS['gnode_semiimplicit']);x=states();y=targets(x,horizon)
    loss,full=m.micro_grad(tf.constant(x),tf.constant(y));part=[]
    for start in [0,2]:part.append(m.micro_grad(tf.constant(x[start:start+2]),tf.constant(y[start:start+2]))[1])
    for a,b,c in zip(full,*part):
        np.testing.assert_allclose(a.numpy(),(.5*(b+c)).numpy(),rtol=3e-4,atol=1e-9)
    assert np.isfinite(float(loss))
    y[:,-1,:,:2]+=.01;_,changed=m.micro_grad(tf.constant(x),tf.constant(y))
    assert any(not np.array_equal(a.numpy(),b.numpy()) for a,b in zip(full,changed))
    result=m.update(x,targets(x,horizon),micro=2);assert np.isfinite(result['gradient_norm'])

def test_pi_native_unclipped_matches_original_a10_initial_and_updates():
    a=StepEngine(OLD_PI['A10']);b=PairEngine(CELLS['pi_ordered_unclipped']);exact(a.arrays(),b.arrays())
    x=tf.constant(states());y=tf.constant(targets(x.numpy())[:,0])
    for _ in range(3):
        ra=a.step(x,y);rb=b.step(x,y);exact(a.arrays(),b.arrays())
        for key in ['data','physics','total']:np.testing.assert_array_equal(ra[key].numpy(),rb[key].numpy())

def test_pi_order_and_clip_do_not_change_initial_weights_or_loss():
    base=PairEngine(CELLS['pi_ordered_unclipped']);x=tf.constant(states());y=tf.constant(targets(x.numpy())[:,0]);raw=base.loss_parts(x,y)
    for name in ['pi_shuffled_unclipped','pi_ordered_clip1','pi_shuffled_clip1']:
        m=PairEngine(CELLS[name]);exact(base.arrays(),m.arrays())
        for key,val in m.loss_parts(x,y).items():np.testing.assert_array_equal(val.numpy(),raw[key].numpy())
        row=m.step(x,y)
        if m.cell.clip_norm:assert float(row['postclip_gradient_norm'])<=1.000001

@pytest.mark.parametrize('name',['pi_ordered_unclipped','pi_shuffled_unclipped','pi_shuffled_clip1'])
def test_pi_sampler_full_epoch_remainder_and_exact_next_epoch(name):
    m=PairEngine(CELLS[name]);sampler=m.sampler();pieces=[sampler.next_indices() for _ in range(1250)]
    assert len(pieces[-1])==192
    np.testing.assert_array_equal(np.sort(np.concatenate(pieces)),np.arange(639680))
    assert np.array_equal(pieces[0],np.arange(512)) is (not m.cell.shuffled)
    other=m.sampler();other.restore(sampler.state());np.testing.assert_array_equal(sampler.next_indices(),other.next_indices())
    assert sampler.state()==other.state()

def test_central_pair_initialization_symmetries_and_internal_force():
    a=PairEngine(CELLS['central_pair_gravity_off']);b=PairEngine(CELLS['central_pair_gravity_scaled']);exact(a.arrays(),b.arrays())
    x=states();pred=a.model(tf.constant(x)).numpy();assert a.model.count_params()==17153
    permutation=[2,0,1]
    np.testing.assert_allclose(a.model(tf.constant(x[:,permutation])).numpy(),pred[:,permutation],rtol=2e-5,atol=2e-6)
    moved=x.copy();moved[...,1:3]+=[.37,-.29];moved[...,3:5]+=[.31,-.23]
    np.testing.assert_allclose(a.model(tf.constant(moved)).numpy(),pred,rtol=2e-5,atol=2e-6)
    rotation=np.array([[0.,-1.],[1.,0.]],np.float32);xr=x.copy();xr[...,1:3]=x[...,1:3]@rotation.T
    np.testing.assert_allclose(a.model(tf.constant(xr)).numpy(),pred@rotation.T,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose((x[...,:1]*pred).sum(1),0.,rtol=0,atol=2e-6)
    y=tf.constant(targets(x)[:,0])
    np.testing.assert_array_equal(a.loss_parts(tf.constant(x),y)['data'].numpy(),b.loss_parts(tf.constant(x),y)['data'].numpy())
    assert a.recipe.physics_weight==0 and b.recipe.physics_weight==.1/882.8257392887597
    for engine in [a,b]:
        row=engine.step(tf.constant(x),y);assert all(np.isfinite(float(v)) for v in row.values())

def test_seed43_propagates_to_native_and_pair_initialization():
    from dataclasses import replace
    for name,constructor in [('gnode_native',GNODEEngine),('pi_shuffled_unclipped',PairEngine),('central_pair_gravity_scaled',PairEngine)]:
        a=constructor(CELLS[name]);cell=replace(CELLS[name],seed=43);b=constructor(cell);c=constructor(cell)
        assert b.recipe.seed==43 and b.recipe.record()['cell']['seed']==43
        exact(b.arrays(),c.arrays())
        assert any(not np.array_equal(x,y) for x,y in zip(a.arrays(),b.arrays()))

def test_nonfinite_raw_gradient_rejected():
    m=PairEngine(CELLS['pi_shuffled_clip1']);x=states();x[0,0,1]=np.nan
    with pytest.raises(tf.errors.InvalidArgumentError):m.step(tf.constant(x),tf.constant(targets(states())[:,0]))
    assert int(m.optimizer.iterations)==0

@pytest.mark.parametrize('name',['pi_shuffled_unclipped','pi_shuffled_clip1','central_pair_gravity_scaled','gnode_semiimplicit'])
def test_exact_model_adam_sampler_progress_resume_and_wrong_cell(tmp_path,name):
    graph=name.startswith('gnode');m=GNODEEngine(CELLS[name]) if graph else PairEngine(CELLS[name])
    sampler=REFERENCE.Cursor() if graph else m.sampler(size=1025)
    x=states();y=targets(x,5) if graph else targets(x)[:,0]
    def update(engine,s):
        if graph:s.position+=1;engine.update(x,y,micro=2)
        else:s.next_indices();engine.step(tf.constant(x),tf.constant(y))
    update(m,sampler);identity={'cell':CELLS[name].record(),'environment':ENVIRONMENT};progress={'updates':1,'norm_sum':3.25,'loss_history':[.1]}
    path=tmp_path/'checkpoint';save_checkpoint(path,m,sampler,identity,progress)
    update(m,sampler)
    clone=GNODEEngine(CELLS[name]) if graph else PairEngine(CELLS[name]);other=REFERENCE.Cursor() if graph else clone.sampler(size=1025)
    assert restore_checkpoint(path,clone,other,identity)==progress
    update(clone,other);exact(m.arrays(),clone.arrays());assert sampler.state()==other.state()
    with pytest.raises(ValueError):restore_checkpoint(path,clone,other,{**identity,'wrong_cell':True})
