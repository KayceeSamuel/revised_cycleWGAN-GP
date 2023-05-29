import pytest
import torch

from cycle_gan_model import CycleGANModel
from options.train_options import TrainOptions
from util import util

# Helper function to get a default options object
def get_default_options():
    options = TrainOptions().parse(['--dataroot', './datasets/facades', '--name', 'facades_label2photo', '--model', 'cycle_gan'])
    options.gpu_ids = []
    options.isTrain = True
    return options

# Helper function to get a CycleGANModel object
def get_model():
    options = get_default_options()
    model = CycleGANModel(options)
    model.real_A = torch.randn((1, options.input_nc, 256, 256))
    model.real_B = torch.randn((1, options.output_nc, 256, 256))
    return model


def test_GAN_loss():
    model = get_model()
    model.forward()

    model.set_requires_grad([model.netD_A, model.netD_B], False)  
    model.optimizer_G.zero_grad()  
    model.backward_G()             

    loss_G = model.loss_G_A + model.loss_G_B + model.loss_cycle_A + model.loss_cycle_B + model.loss_idt_A + model.loss_idt_B

    assert loss_G is not None
    assert torch.isfinite(loss_G).all()


def test_discriminator_loss():
    model = get_model()
    model.forward()

    model.set_requires_grad([model.netD_A, model.netD_B], True)
    model.optimizer_D.zero_grad()

    fake_B = model.fake_B_pool.query(model.fake_B)
    loss_D_A = model.backward_D_basic(model.netD_A, model.real_B, fake_B)
    fake_A = model.fake_A_pool.query(model.fake_A)
    loss_D_B = model.backward_D_basic(model.netD_B, model.real_A, fake_A)

    assert loss_D_A is not None
    assert torch.isfinite(loss_D_A).all()
    assert loss_D_B is not None
    assert torch.isfinite(loss_D_B).all()


def test_identity_loss():
    model = get_model()
    model.forward()

    model.set_requires_grad([model.netD_A, model.netD_B], False)  
    model.optimizer_G.zero_grad()  
    model.backward_G()             

    assert model.loss_idt_A is not None
    assert torch.isfinite(model.loss_idt_A).all()
    assert model.loss_idt_B is not None
    assert torch.isfinite(model.loss_idt_B).all()