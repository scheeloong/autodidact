from collections import defaultdict
from itertools import count
import numpy as np

from .tracer import trace, Node
from .util import toposort

def make_vjp(fun, x):
    '''
    Given a function 
    and
    The value for that function
    Vector Jacobian Products
    '''
    print("Inside make_vjp")
    # Create a new root
    start_node = Node.new_root()
    print("Start node is", start_node)
    # 
    end_value, end_node = trace(start_node, fun, x)
    if end_node is None:
        def vjp(g): return np.zeros_like(x)
    else:
        def vjp(g): return backward_pass(g, end_node)
    print("End node is", end_node)
    return vjp, end_value

def backward_pass(g, end_node):
    print("Inside backward_pass")
    outgrads = {end_node: g}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        fun, value, args, kwargs, argnums = node.recipe
        for argnum, parent in zip(argnums, node.parents):
            vjp = primitive_vjps[fun][argnum]
            parent_grad = vjp(outgrad, value, *args, **kwargs)
            outgrads[parent] = add_outgrads(outgrads.get(parent), parent_grad)
    return outgrad

def add_outgrads(prev_g, g):
    if prev_g is None:
        return g
    return prev_g + g

primitive_vjps = defaultdict(dict)
def defvjp(fun, *vjps, **kwargs):
    argnums = kwargs.get('argnums', count())
    for argnum, vjp in zip(argnums, vjps):
        primitive_vjps[fun][argnum] = vjp
