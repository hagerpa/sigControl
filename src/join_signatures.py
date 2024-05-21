import iisignature

import torch


def generate_joinsig_function(d, N, space='log'):
    """
    Creats a torch autograd function around the `iisignature.logsigjoin` resp. `iisignature.sigjoin` funciton.
    :param d: dimension of the input path.
    :param N: signature truncation level.
    :param space: 'log' for log signature. 'sig' for actual signature.
    :return: Torch autograd-function with input (X, D) where X is a (log-)signature and D is a path increment.
    """
    if space == 'log':
        join = iisignature.logsigjoin
        join_backprop = iisignature.logsigjoinbackprop
        args = (iisignature.prepare(d, N, 'O'),)
        sig_dim = iisignature.logsiglength(d, N)
    elif space == 'sig':
        join = iisignature.sigjoin
        join_backprop = iisignature.sigjoinbackprop
        args = (N,)
        sig_dim = iisignature.siglength(d, N)
    else:
        sig_dim = None
        assert False,"Chose either 'log' or 'sig' for the 'space' parameter."

    class sig_fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X, D):
            result = join(X, D, *args)
            ctx.save_for_backward(X, D)
            if ctx.needs_input_grad[0]:
                return torch.from_numpy(result).type(torch.float64)
            else:
                return torch.tensor(result, dtype=torch.float64, requires_grad=False)

        @staticmethod
        def backward(ctx, grad_output):
            X, D = ctx.saved_tensors
            dX, dD = join_backprop(grad_output, X, D, *args)
            return torch.from_numpy(dX).type(torch.float64), torch.from_numpy(dD).type(torch.float64)

    return sig_fn, sig_dim
