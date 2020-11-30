#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torch.nn.functional as thf

import contextlib


class Interp1(th.autograd.Function):

    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation linke :func:`numpy.interp` on the GPU for Pyth.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = th.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1] and (v['x'].shape[0] == v['y'].shape[0] or v['x'].shape[0] == 1 or v['y'].shape[
                0] == 1)), ("x and y must have the same number of columns, and either " "the same number of row or one of them having only one " "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1) and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0] * shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = th.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        ind = th.searchsorted(v['x'].contiguous(), v['xnew'].contiguous(), out=None)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = th.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return th.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with th.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                (v['y'][:, 1:] - v['y'][:, :-1]) /
                (eps + (v['x'][:, 1:] - v['x'][:, :-1]))
            )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes') * (
                v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = th.autograd.grad(
            ctx.saved_tensors[0],
            [i for i in inputs if i is not None],
            grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    r"""Down/up samples the input to either the given :attr:`size` or the given
        :attr:`scale_factor`

        The algorithm used for interpolation is determined by :attr:`mode`.

        Currently temporal, spatial and volumetric sampling are supported, i.e.
        expected inputs are 3-D, 4-D or 5-D in shape.

        The input dimensions are interpreted in the form:
        `mini-batch x channels x [optional depth] x [optional height] x width`.

        The modes available for resizing are: `nearest`, `linear` (3D-only),
        `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

        Args:
            input (Tensor): the input tensor
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
                output spatial size.
            scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
            mode (str): algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
            align_corners (bool, optional): Geometrically, we consider the pixels of the
                input and output as squares rather than points.
                If set to ``True``, the input and output tensors are aligned by the
                center points of their corner pixels, preserving the values at the corner pixels.
                If set to ``False``, the input and output tensors are aligned by the corner
                points of their corner pixels, and the interpolation uses edge value padding
                for out-of-boundary values, making this operation *independent* of input size
                when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
                is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
                Default: ``False``
            recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
                interpolation calculation.  When `scale_factor` is passed as a parameter, it is used
                to compute the `output_size`.  If `recompute_scale_factor` is ```False`` or not specified,
                the passed-in `scale_factor` will be used in the interpolation computation.
                Otherwise, a new `scale_factor` will be computed based on the output and input sizes for
                use in the interpolation computation (i.e. the computation will be identical to if the computed
                `output_size` were passed-in explicitly).  Note that when `scale_factor` is floating-point,
                the recomputed scale_factor may differ from the one passed in due to rounding and precision
                issues.

        .. note::
            With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
            negative values or values greater than 255 for images.
            Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
            when displaying the image.

        .. warning::
            With ``align_corners = True``, the linearly interpolating modes
            (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
            output and input pixels, and thus the output values can depend on the
            input size. This was the default behavior for these modes up to version
            0.3.1. Since then, the default behavior is ``align_corners = False``.
            See :class:`~th.nn.Upsample` for concrete examples on how this
            affects the outputs.

        .. warning::
            When scale_factor is specified, if recompute_scale_factor=True,
            scale_factor is used to compute the output_size which will then
            be used to infer new scales for the interpolation.
            The default behavior for recompute_scale_factor changed to False
            in 1.6.0, and scale_factor is used in the interpolation
            calculation.

        Note:
            When using the CUDA backend, this operation may induce nondeterministic
            behaviour in its backward pass that is not easily switched off.
            Please see the notes on :doc:`/notes/randomness` for background.
        """

    return thf.interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor)


def interpolatec(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    r"""Down/up samples the input to either the given :attr:`size` or the given
        :attr:`scale_factor`

        The algorithm used for complex valued interpolation is determined by :attr:`mode`.

        Currently temporal, spatial and volumetric sampling are supported, i.e.
        expected inputs are 3-D, 4-D or 5-D in shape.

        The input dimensions are interpreted in the form:
        `mini-batch x [optional channels] x [optional height] x width x 2`.

        The modes available for resizing are: `nearest`, `linear` (3D-only),
        `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

        Args:
            input (Tensor): the input tensor
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
                output spatial size.
            scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
            mode (str): algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
            align_corners (bool, optional): Geometrically, we consider the pixels of the
                input and output as squares rather than points.
                If set to ``True``, the input and output tensors are aligned by the
                center points of their corner pixels, preserving the values at the corner pixels.
                If set to ``False``, the input and output tensors are aligned by the corner
                points of their corner pixels, and the interpolation uses edge value padding
                for out-of-boundary values, making this operation *independent* of input size
                when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
                is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
                Default: ``False``
            recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
                interpolation calculation.  When `scale_factor` is passed as a parameter, it is used
                to compute the `output_size`.  If `recompute_scale_factor` is ```False`` or not specified,
                the passed-in `scale_factor` will be used in the interpolation computation.
                Otherwise, a new `scale_factor` will be computed based on the output and input sizes for
                use in the interpolation computation (i.e. the computation will be identical to if the computed
                `output_size` were passed-in explicitly).  Note that when `scale_factor` is floating-point,
                the recomputed scale_factor may differ from the one passed in due to rounding and precision
                issues.

        .. note::
            With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
            negative values or values greater than 255 for images.
            Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
            when displaying the image.

        .. warning::
            With ``align_corners = True``, the linearly interpolating modes
            (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
            output and input pixels, and thus the output values can depend on the
            input size. This was the default behavior for these modes up to version
            0.3.1. Since then, the default behavior is ``align_corners = False``.
            See :class:`~th.nn.Upsample` for concrete examples on how this
            affects the outputs.

        .. warning::
            When scale_factor is specified, if recompute_scale_factor=True,
            scale_factor is used to compute the output_size which will then
            be used to infer new scales for the interpolation.
            The default behavior for recompute_scale_factor changed to False
            in 1.6.0, and scale_factor is used in the interpolation
            calculation.

        Note:
            When using the CUDA backend, this operation may induce nondeterministic
            behaviour in its backward pass that is not easily switched off.
            Please see the notes on :doc:`/notes/randomness` for background.
        """

    dim0 = list(range(input.dim()))
    dim = dim0.copy()
    dim.insert(1, dim[-1])
    dim.pop()

    input = input.permute(dim)

    dim0[1:-1] = dim0[2:-1]
    dim0.append(1)

    return thf.interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor).permute(dim0)


if __name__ == "__main__":

    import torch
    import matplotlib.pyplot as plt
    import time
    import numpy as np
    # defining the number of tests
    ntests = 2

    # problem dimensions
    D = 1000
    Dnew = 1
    N = 100
    P = 30

    yq_gpu = None
    yq_cpu = None
    for ntest in range(ntests):
        # draw the data
        x = torch.rand(D, N) * 10000
        x = x.sort(dim=1)[0]

        y = torch.linspace(0, 1000, D * N).view(D, -1)
        y -= y[:, 0, None]

        xnew = torch.rand(Dnew, P) * 10000

        print('Solving %d interpolation problems: '
              'each with %d observations and %d desired values' % (D, N, P))

        # calling the cpu version
        t0_cpu = time.time()
        yq_cpu = Interp1()(x, y, xnew, yq_cpu)
        t1_cpu = time.time()

        display_str = 'CPU: %0.3fms, ' % ((t1_cpu - t0_cpu) * 1000)

        if torch.cuda.is_available():
            x = x.to('cuda')
            y = y.to('cuda')
            xnew = xnew.to('cuda')

            # launching the cuda version
            t0 = time.time()
            yq_gpu = Interp1()(x, y, xnew, yq_gpu)
            t1 = time.time()

            # compute the difference between both
            error = torch.norm(
                yq_cpu - yq_gpu.to('cpu')) / torch.norm(yq_cpu) * 100.

            display_str += 'GPU: %0.3fms, error: %f%%.' % (
                (t1 - t0) * 1000, error)
        print(display_str)

    if torch.cuda.is_available():
        # for the last test, plot the result for the first 10 dimensions max
        d_plot = min(D, 10)
        x = x[:d_plot].cpu().numpy()
        y = y[:d_plot].cpu().numpy()
        xnew = xnew[:d_plot].cpu().numpy()
        yq_cpu = yq_cpu[:d_plot].cpu().numpy()
        yq_gpu = yq_gpu[:d_plot].cpu().numpy()

        plt.plot(x.T, y.T, '-',
                 xnew.T, yq_gpu.T, 'o',
                 xnew.T, yq_cpu.T, 'x')
        not_close = np.nonzero(np.invert(np.isclose(yq_gpu, yq_cpu)))
        if not_close[0].size:
            plt.scatter(xnew[not_close].T, yq_cpu[not_close].T,
                        edgecolors='r', s=100, facecolors='none')
        plt.grid(True)
        plt.show()
