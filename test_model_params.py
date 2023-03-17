from torchstat import stat
from thop import profile
import torch.nn as nn
import pdb
from models.PC2D import *
from models.PCC_util import *
from models.depth_utils import *
from models.B_Trans_PCC import Encoder_PointTrans

def test_AE(latent_size):
    model = LatentAE(latent_size)
    input = torch.randn(1, latent_size, 1)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

def test_Trans(latent_size):
    model = Encoder_PointTrans(latent_size)
    input = torch.randn(1, 2048, 3)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

def test_NGS(latent_size):
    model = NGSEncoder(latent_size)
    input = torch.randn(1, 2048, 3)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

def test_PCT(latent_size):
    model = PCTEncoder(latent_size)
    input = torch.randn(1, 2048, 3)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

def test_PN(latent_size):
    model = PNEncoder(latent_size)
    input = torch.randn(1, 2048, 3)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

def test_PNPP(latent_size):
    model = PNPPEncoder(latent_size)
    input = torch.randn(1, 2048, 3)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

def test_Seg(latent_size):
    model = EnBEncoder(latent_size)
    input = torch.randn(1, 2048, 3)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

def test_Fusion(latent_size):
    model = nn.Linear(512, 256)
    input = torch.randn(1, 512)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

def test_Decoder(latent_size):
    model = PCCDecoder(256, 2048)
    input = torch.randn(1, 256)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

test_Trans(256)
