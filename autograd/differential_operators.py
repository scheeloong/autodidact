"""Convenience functions built on top of `make_vjp`."""

import numpy as np

from .core import make_vjp
from .util import subval

def grad(fun, argnum=0):
    def gradfun(*args, **kwargs):
        print("Args are", args)
        print("Kwargs are", kwargs)
        # Define a function that takes in parameter x
        # and runs fun(*subval(args, argnum, x), **kwargs)
        unary_fun = lambda x: fun(*subval(args, argnum, x), **kwargs)
        print("Unary fun are", unary_fun)
        # Make VectorJacobianProduct for the unary function
        vjp, ans = make_vjp(unary_fun, args[argnum])
        print("VJP are", vjp)
        print("ANS are", ans)
        return vjp(np.ones_like(ans))
    return gradfun
