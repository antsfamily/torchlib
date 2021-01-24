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
from .misc.sampling import sample_tensor, split_tensor

from .module.loss.contrast import ContrastLoss
from .module.loss.entropy import EntropyLoss
from .module.loss.norm import FrobeniusLoss, LogFrobeniusLoss
from .module.loss.perceptual import RandomProjectionLoss
from .module.loss.retrieval import DiceLoss, JaccardLoss, F1Loss
from .module.loss.semantic import EdgeLoss
# from .module.loss.segmentation import SoftDiceLoss, FocalLoss, BinaryCrossEntropyLoss, ContourLoss, EdgeAwareLoss
from .module.loss.ssims import SSIMLoss, MSSSIMLoss, StructureLoss
from .module.loss.variation import TotalVariation
from .module.layers.cnnsize import ConvSize1d, ConvTransposeSize1d, PoolSize1d, UnPoolSize1d, ConvSize2d, ConvTransposeSize2d, PoolSize2d, UnPoolSize2d
from .module.layers.edge import EdgeDetector, EdgeFeatureExtractor
from .module.layers.pool import MeanSquarePool2d, PnormPool2d

from .module.layers.complex_functions import complex_relu, complex_leaky_relu, complex_max_pool2d, complex_dropout, complex_dropout2d, complex_upsample
from .module.layers.complex_layers import ComplexSequential, ComplexMaxPool2d, ComplexMaxPool1d, ComplexDropout,  ComplexDropout2d, ComplexReLU, ComplexLeakyReLU, ComplexConvTranspose2d, ComplexConv2d, ComplexConvTranspose1d, ComplexConv1d, ComplexLinear, ComplexUpsample, NaiveComplexBatchNorm1d, NaiveComplexBatchNorm2d, NaiveComplexBatchNorm1d, ComplexBatchNorm2d, ComplexBatchNorm1d, ComplexConv1, ComplexMaxPool1, ComplexConv2, ComplexMaxPool2
from .module.layers.fft_layers import FFTLayer1d
from .module.layers.convolution import FFTConv1, Conv1, Conv2, MaxPool1, MaxPool2

from .module.misc.transform import Standardization

from .dsp.filters import sobelfilter
from .dsp.kernels import *
from .dsp.ffts import padfft, fft, ifft, fftfreq, fftshift, ifftshift
from .dsp.convolution import fftconv1
from .dsp.interpolation import interpolate, interpolatec, Interp1
from .dsp.normalsignals import rect, chirp

from .spl import voptimizer
from .spl import spfunction

from .optim.learning_rate import gammalr, LrFinder
from .optim.lr_scheduler import DoubleGaussianKernelLR
from .optim.save_load import device_transfer

from .summary.loss_log import LossLog

from .diagnose.plotgradflow import plot_gradflow_v1, plot_gradflow_v2

from .utils.const import *
from .utils.ios import loadmat, savemat, loadh5, saveh5
from .utils.file import listxfile, pathjoin, fileparts, readtxt
from .utils.randomfunc import setseed, randgrid, randgrid2d, randperm, randperm2d


