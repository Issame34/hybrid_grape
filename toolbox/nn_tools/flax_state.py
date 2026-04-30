import jax
import flax.linen as nn
from flax.training import train_state
import os
import orbax.checkpoint as ocp


def create_flax_state(key, nn_call, dummy_inp, optimizer, *, print_summary=True):
    """
    Create a FLAX training state for neural support
    """
    class NeuralModule(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn_call(x)
                   
    neural_module = NeuralModule() 
    params = neural_module.init(key, dummy_inp)['params']
    state = train_state.TrainState.create(
        apply_fn=neural_module.apply,
        params=params,
        tx=optimizer
    )
    
    if print_summary:
        print(neural_module.tabulate(jax.random.key(0), dummy_inp))
        
    return state


def save_flax_state(fld, state):
    abs_path = os.path.abspath(fld)
    orbax_path = ocp.test_utils.erase_and_create_empty(abs_path) / "1"
    
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(
        orbax_path,
        args=ocp.args.StandardSave(state)
    )
    ckptr.wait_until_finished()
    return None


def load_flax_state(fld, state):
    abs_path = os.path.abspath(fld)
    orbax_path = ocp.test_utils.epath.gpath.PosixGPath(abs_path) / "1"
    
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored_state = ckptr.restore(
        orbax_path, args=ocp.args.StandardRestore(state)
    )
    return restored_state