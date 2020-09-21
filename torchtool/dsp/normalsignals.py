import torch as th
from torchtool.utils.const import PI


def rect(x):
    r"""
    Rectangle function:
        rect(x) = {1, if |x|<= 0.5; 0, otherwise}
    """
    # return hs(x + 0.5) * ihs(x - 0.5)
    # return th.where(th.abs(x) > 0.5, 0., 1.0)
    y = x.clone()
    y[x < -0.5] = 0.
    y[x > 0.5] = 0.
    return y


def chirp(t, T, Kr):
    r"""
    Create a chirp signal :
        S_{tx}(t) = rect(t/T) * exp(1j*pi*Kr*t^2)
    """
    return rect(t / T) * th.exp(1j * PI * Kr * t**2)
