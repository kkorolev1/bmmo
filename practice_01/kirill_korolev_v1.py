# There should be no main() in this file!!! 
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1, 
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,) 
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

# In variant 1 the following functions are required:
import numpy as np
from scipy.stats import binom, poisson


def pa(params, model):
    vals = np.arange(params['amin'], params['amax'] + 1)
    return np.full((len(vals),), 1 / (params['amax'] - params['amin'] + 1)), vals

def pb(params, model):
    vals = np.arange(params['bmin'], params['bmax'] + 1)
    return np.full((len(vals),), 1 / (params['bmax'] - params['bmin'] + 1)), vals

def pc(params, model):
    pa_probs, avals = pa(params, model)
    pb_probs, bvals = pb(params, model)
    pc_ab_probs, cvals = pc_ab(avals, bvals, params, model)
    probs = ((pc_ab_probs * pb_probs).sum(axis=-1) * pa_probs).sum(axis=-1)
    return probs, cvals

def pd(params, model):
    pc_probs, cvals = pc(params, model)
    pd_c_probs, dvals = pd_c(cvals, params, model)
    pd_c_probs = pd_c_probs
    probs = (pd_c_probs * pc_probs).sum(axis=-1)
    return probs, dvals

def pc_a(a, params, model):
    pc_ab_probs, cvals = pc_ab(a, np.arange(params['bmin'], params['bmax'] + 1),
                               params, model)
    pb_probs, _ = pb(params, model)
    probs = (pc_ab_probs * pb_probs).sum(axis=-1)
    return probs, cvals

def pc_b(b, params, model):
    pc_ab_probs, cvals = pc_ab(np.arange(params['amin'], params['amax'] + 1), b,
                               params, model)
    pa_probs, _ = pa(params, model)
    probs = (pc_ab_probs * pa_probs[None,:,None]).sum(axis=1)
    return probs, cvals

def pd_c(c, params, model):
    cmax = (params['amax'] + params['bmax'])
    vals = np.arange(2 * cmax + 1)
    probs = binom.pmf(vals[:, None] - c[None, :], n=c, p=params['p3'])
    return probs, vals

def pc_d(d, params, model):
    pc_probs, cvals = pc(params, model)
    pd_c_probs, _ = pd_c(cvals, params, model)
    numero = (pd_c_probs[d, :] * pc_probs).transpose(1, 0)
    probs = numero / numero.sum(axis=0)
    return probs, cvals

def pc_ab(a, b, params, model):
    if model == 1:
        cvals = np.arange(0, params['amax'] + params['bmax'] + 1)
        
        bin_a = binom.pmf(cvals[:, None], n=a, p=params['p1'])
        bin_b = binom.pmf(cvals[:, None], n=b, p=params['p2'])
        
        probs = np.zeros((len(cvals), len(a), len(b)), dtype=np.float64)
        for i in range(len(cvals)):
            probs[i, :, :] = bin_a[:i + 1].T @ bin_b[:i + 1][::-1]

        return probs, cvals
    elif model == 2:
        cvals = np.arange(0, params['amax'] + params['bmax'] + 1)
        probs = poisson.pmf(cvals[:, None, None], a[:, None] * params['p1'] + b[None, :] * params['p2'])
        return probs, cvals
    else:
        raise RuntimeError(f'Unknown model={model}')
    
def pc_abd(a, b, d, params, model):
    pc_ab_probs, cvals = pc_ab(a, b, params, model)
    pd_c_probs, _ = pd_c(cvals, params, model)
    pd_c_probs = pd_c_probs[d, :]
    
    tensor_prod = np.tensordot(pd_c_probs, pc_ab_probs, axes=0)
    numero = np.diagonal(tensor_prod, axis1=1, axis2=2).transpose(3, 1, 2, 0)
    probs = numero / numero.sum(axis=0)
    return probs, cvals

params = {'amin': 75, 'amax': 90, 'bmin': 500, 'bmax': 600,
              'p1': 0.1, 'p2': 0.01, 'p3': 0.3}
probs, vals = pc_a(np.array([82]), params, 1)
print((probs * vals).sum())
