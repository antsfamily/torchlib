from __future__ import absolute_import

from .base.ops import cat, concat2
from .base.mathops import nextpow2, prevpow2, ebemulcc, mmcc, matmulcc
from .base.arrayops import sl, cut


from .evaluation.classification import accuracy


from .evaluation.entropy import entropy
from .evaluation.retrieval import true_positive, true_negative, \
    false_positive, false_negative, \
    precision, recall, sensitivity, selectivity, fmeasure
from .evaluation.similarity import jaccard_index, dice_coeff
from .evaluation.ssims import create_window, _ssim, ssim, msssim


from .misc.transform import standardization, scale
from .misc.mapping_operation import mapping


from .dsp.filters import sobelfilter
from .dsp.kernels import *
from .dsp.ffts import padfft, fft, ifft, fftfreq, fftshift, ifftshift
from .dsp.convolution import fftconv1
from .dsp.interpolation import interpolate, interpolatec, Interp1
from .dsp.normalsignals import rect, chirp

from .spl import voptimizer
from .spl import spfunction

from .optim.learning_rate import LrFinder
from .summary.loss_log import LossLog

from .diagnose.plotgradflow import plot_gradflow_v1, plot_gradflow_v2

from .utils.const import *





from .nn.module.cnnsize import ConvSize1d, ConvTransposeSize1d, PoolSize1d, UnPoolSize1d, ConvSize2d, ConvTransposeSize2d, PoolSize2d, UnPoolSize2d
from .nn.module.edge import EdgeDetector, EdgeFeatureExtractor
from .nn.module.pool import MeanSquarePool2d, PnormPool2d
from .nn.loss.retrieval import DiceLoss, JaccardLoss, IridescentLoss, F1Loss
from .nn.loss.ssims import SSIMLoss, MSSSIMLoss, StructureLoss
from .nn.loss.semantic import EdgeLoss
from .nn.loss.entropy import ShannonEntropy, NaturalEntropy
from .nn.loss.variation import TotalVariation
from .nn.loss.contrast import ContrastLoss
from .nn.loss.norm import Frobenius
from .nn.module.complex_functions import complex_relu, complex_leaky_relu, complex_max_pool2d, complex_dropout, complex_dropout2d, complex_upsample
from .nn.module.complex_layers import ComplexSequential, ComplexMaxPool2d, ComplexMaxPool1d, ComplexDropout,  ComplexDropout2d, ComplexReLU, ComplexLeakyReLU, ComplexConvTranspose2d, ComplexConv2d, ComplexConvTranspose1d, ComplexConv1d, ComplexLinear, ComplexUpsample, NaiveComplexBatchNorm1d, NaiveComplexBatchNorm2d, NaiveComplexBatchNorm1d, ComplexBatchNorm2d, ComplexBatchNorm1d, ComplexConv1, ComplexMaxPool1, ComplexConv2, ComplexMaxPool2
from .nn.module.fft_layers import FFTLayer1d
from .nn.module.convolution import FFTConv1, Conv1, Conv2, MaxPool1, MaxPool2


