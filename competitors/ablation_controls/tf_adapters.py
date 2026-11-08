"""TF-only adapters. Configure the reviewed runtime BEFORE importing this module."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import types
import numpy as np
import tensorflow as tf
from competitors.gnode_pignn_training.engine import StepEngine
from competitors.gnode_pignn_training.data import PairSampler
from competitors.pignn_ablation.recipes import CELLS as PI_CELLS
from .recipes import validate_cell
ROOT=Path(__file__).resolve().parents[2]
ACCELERATION_MEAN_SQUARE=882.8257392887597

@dataclass(frozen=True)
class _MultiRecipe:
    kind: str = 'gnode'
    seed: int = 42
    dt: float = .001
    learning_rate: float = .0005
    adam_epsilon: float = 1e-7
    batch_size: int = 64
    physics_weight: float = 0.
    def record(self):
        return {**asdict(self), 'protocol':'gnode-staged-multistep-v1', 'parameters':83714,
                'loss':'mean over future steps, bodies, x/y/vx/vy squared residuals; no normalization',
                'horizons':[1,5,20,50,100], 'updates_per_horizon':500, 'updates':2500,
                'micro_batch_size':16, 'bptt':'full; no stop_gradient',
                'clipping':None,'weight_decay':None,'scheduler':None,
                'selection':'minimum eligible full1999-future VAL80 position MAE every100updates; earlier ties',
                'inference_position':'expanded_constant_acceleration', 'training_position':'average_velocity',
                'existing100_read':False,
                'comparison_caveat':'Different budget/batch/order from historical 125000-update GNODE; not isolated C11 effect'}

class _Cursor:
    def __init__(self):
        self.position = 0
        self.rng = np.random.default_rng(42)
    def state(self):
        return {'position':self.position,'rng':self.rng.bit_generator.state,'order':'identity2500'}
    def restore(self, state):
        if state['order'] != 'identity2500' or not 0 <= state['position'] <= 2500:
            raise ValueError('invalid schedule identity/cursor')
        self.position = state['position']; self.rng.bit_generator.state = state['rng']

class _MultiEngine(StepEngine):
    def __init__(self):
        super().__init__(_MultiRecipe())
        self.micro_grad = tf.function(self._micro_grad, input_signature=[
            tf.TensorSpec([None,3,5],tf.float32),tf.TensorSpec([None,None,3,4],tf.float32)],jit_compile=False)
        self.apply = tf.function(self._apply, jit_compile=False)
    def loss_multistep(self, x, y):
        horizon = tf.shape(y)[1]
        def body(k, state, loss):
            out = self.model(state, training=True)
            loss += tf.reduce_mean(tf.square(out-y[:,k]))
            return k+1, tf.concat([state[...,:1],out],axis=-1), loss
        _, _, total = tf.while_loop(lambda k,s,l:k<horizon, body,
                                   (tf.constant(0),x,tf.constant(0.,tf.float32)),parallel_iterations=1)
        return total/tf.cast(horizon,tf.float32)
    def _micro_grad(self,x,y):
        with tf.GradientTape() as tape: loss = self.loss_multistep(x,y)
        grads = tape.gradient(loss,self.model.trainable_variables)
        if any(g is None for g in grads): raise ValueError('disconnected gradient')
        return tf.debugging.check_numerics(loss,'nonfinite loss'), [tf.debugging.check_numerics(g,'nonfinite gradient') for g in grads]
    def _apply(self,grads):
        checked = [tf.debugging.check_numerics(g,'nonfinite accumulated gradient') for g in grads]
        self.optimizer.apply_gradients(zip(checked,self.model.trainable_variables,strict=True))
        for v in [*self.model.variables,*self.optimizer.variables]:
            if v.dtype != 'int64': tf.debugging.check_numerics(v,'nonfinite post-update state')
        return tf.linalg.global_norm(checked)
    def update(self,x,y,micro=16):
        if len(x) % micro: raise ValueError('equal microbatch sizes required')
        grads = [tf.zeros_like(v) for v in self.model.trainable_variables]
        loss = 0.; weight = micro/len(x)
        for start in range(0,len(x),micro):
            l,g = self.micro_grad(tf.constant(x[start:start+micro]),tf.constant(y[start:start+micro]))
            loss += float(l)*weight
            grads = [a+b*weight for a,b in zip(grads,g,strict=True)]
        norm = float(self.apply(grads))
        return {'loss':loss,'gradient_norm':norm}

def _assigned(states,bank,unit):
    h = int(bank['horizon'][unit]); ids = bank['trajectory_index'][unit]; starts = bank['start_frame'][unit]
    x = np.stack([states[int(i),int(t)] for i,t in zip(ids,starts,strict=True)])[...,[4,0,1,2,3]].astype(np.float32)
    y = np.stack([states[int(i),int(t)+1:int(t)+1+h,:,:4] for i,t in zip(ids,starts,strict=True)]).astype(np.float32)
    assert y.shape == (64,h,3,4)
    return x,y

REFERENCE = types.SimpleNamespace(MultiRecipe=_MultiRecipe, Cursor=_Cursor, MultiEngine=_MultiEngine, assigned=_assigned)

class SemiImplicitODE(tf.keras.layers.Layer):
    def __init__(self,dt,**kwargs):super().__init__(**kwargs);self.dt=dt
    def call(self,state,acceleration):
        velocity=state[...,3:5]+self.dt*acceleration
        return tf.concat([state[...,1:3]+self.dt*velocity,velocity],axis=-1)

class GNODERecipe:
    def __init__(self,cell):self.cell=cell;self.base=REFERENCE.MultiRecipe(seed=cell.seed)
    def __getattr__(self,name):return getattr(self.base,name)
    def record(self):
        base=self.base.record()
        if self.cell.update=='semiimplicit':base.update(training_position='semiimplicit',inference_position='semiimplicit')
        return {**base,'protocol':'next-study-gnode-v1','cell':self.cell.record()}

class GNODEEngine(REFERENCE.MultiEngine):
    def __init__(self,cell):
        validate_cell(cell)
        if cell.track!='gnode':raise ValueError('not GNODE')
        StepEngine.__init__(self,REFERENCE.MultiRecipe(seed=cell.seed))
        self.micro_grad=tf.function(self._micro_grad,input_signature=[tf.TensorSpec([None,3,5],tf.float32),tf.TensorSpec([None,None,3,4],tf.float32)],jit_compile=False)
        self.apply=tf.function(self._apply,jit_compile=False)
        original=self.model
        if cell.update=='semiimplicit':
            qv=SemiImplicitODE(self.recipe.dt,name='ode')(original.inputs[0],original.get_layer('gnn').output)
            self.model=tf.keras.Model(original.inputs,qv)
            if [id(v) for v in original.trainable_variables]!=[id(v) for v in self.model.trainable_variables]:raise ValueError('optimizer variable order changed')
        self.recipe=GNODERecipe(cell)
    def inference_view(self):
        if self.recipe.cell.update=='semiimplicit':return self.model
        return super().inference_view()

class PairRecipe:
    """Preserve native A10 fields; bind ordering/clip/architecture/physics to metadata."""
    def __init__(self,cell):
        self.cell=cell
        weight=.1 if cell.gravity=='raw' else .1/ACCELERATION_MEAN_SQUARE if cell.gravity=='scaled' else 0.
        self.base=replace(PI_CELLS['A10'],model_id=cell.name,seed=cell.seed,train_position='semiimplicit' if cell.architecture=='central_pair' else 'explicit',physics_weight=weight)
    def __getattr__(self,name):return getattr(self.base,name)
    def record(self):
        return {**self.base.record(),'protocol':'next-study-pi-v1','cell':self.cell.record(),'shuffle':self.cell.shuffled,
            'clipping':self.cell.clip_norm,'architecture':self.cell.architecture,
            'normalization':'force mean-square denominator882.8257392887597' if self.cell.gravity=='scaled' else 'none'}

class CentralPair(tf.keras.layers.Layer):
    """Symmetric scalar pair law times relative displacement; no velocity dependence.

    a_i=sum_j m_j*(q_j-q_i)*f(r2,m_i+m_j,m_i*m_j). Sum m_i*a_i=0 up to rounding.
    Parameter count is not matched to the native PI-GNN. No explicit energy readout.
    """
    def __init__(self,**kw):
        super().__init__(**kw);self.scalar=tf.keras.Sequential([tf.keras.layers.Dense(128,activation='swish'),tf.keras.layers.Dense(128,activation='swish'),tf.keras.layers.Dense(1)])
    def call(self,state):
        q=state[...,1:3];m=state[...,:1];d=q[:,None,:,:]-q[:,:,None,:]
        mi=m[:,:,None,:];mj=m[:,None,:,:]
        mass_sum=mi+mj;mass_product=mi*mj
        features=tf.concat([tf.reduce_sum(d*d,axis=-1,keepdims=True),mass_sum,mass_product],axis=-1)
        scalar=self.scalar(features);n=tf.shape(q)[1]
        mask=(1.-tf.eye(n,dtype=state.dtype))[None,:,:,None]
        return tf.reduce_sum(mj*d*scalar*mask,axis=2)

class PairEngine(StepEngine):
    def __init__(self,cell):
        validate_cell(cell)
        if cell.track not in {'pi_order','central_pair'}:raise ValueError('not pair-engine cell')
        self.cell=cell;recipe=PairRecipe(cell)
        if cell.architecture=='native':super().__init__(recipe)
        else:
            self.recipe=recipe;tf.keras.utils.set_random_seed(recipe.seed)
            inputs=tf.keras.Input(shape=(None,5),dtype='float32')
            self.model=tf.keras.Model(inputs,CentralPair(name='central_pair')(inputs))
            self.optimizer=tf.keras.optimizers.Adam(learning_rate=recipe.learning_rate,beta_1=.9,beta_2=.999,epsilon=recipe.adam_epsilon,amsgrad=False)
            self.optimizer.build(self.model.trainable_variables);self.mse=tf.keras.losses.MeanSquaredError()
            self.step=tf.function(self._step,input_signature=[tf.TensorSpec([None,3,5],tf.float32),tf.TensorSpec([None,3,4],tf.float32)],jit_compile=False)
    def sampler(self,size=639680):return PairSampler(size,self.recipe.batch_size,self.cell.shuffled,self.recipe.seed)
    def _step(self,x,y):
        with tf.GradientTape() as tape:losses=self.loss_parts(x,y)
        gradients=tape.gradient(losses['total'],self.model.trainable_variables)
        if any(g is None for g in gradients):raise ValueError('disconnected gradient')
        gradients=[tf.debugging.check_numerics(g,'nonfinite raw gradient') for g in gradients]
        total=tf.debugging.check_numerics(losses['total'],'nonfinite loss')
        norm=tf.linalg.global_norm(gradients)
        if self.cell.clip_norm is not None:gradients,_=tf.clip_by_global_norm(gradients,self.cell.clip_norm)
        with tf.control_dependencies([total,*gradients]):self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables,strict=True))
        for variable in [*self.model.variables,*self.optimizer.variables]:
            if variable.dtype!='int64':tf.debugging.check_numerics(variable,'nonfinite post-update state')
        return {**losses,'gradient_norm':norm,'postclip_gradient_norm':tf.linalg.global_norm(gradients),
            'clipped':tf.cast(norm>self.cell.clip_norm,tf.float32) if self.cell.clip_norm is not None else tf.constant(0.,tf.float32)}
