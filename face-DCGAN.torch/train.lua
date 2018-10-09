--[[ Trains the Deep Convolutional Generative Network (DCGAN). 

Please see https://arxiv.org/abs/1511.06434.pdf 

Usage: th train.lua --help ]]

require 'cunn'
require 'optim'
require 'cutorch'
require 'paths'

-- set defualt tensor type float
torch.setdefaulttensortype('torch.FloatTensor')

-- local modules
local Utils = require('utils')
local DCGAN = require('model')

-- command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text('Trains a Deep Convolutional Generative Network (DCGAN)')
cmd:text()
cmd:text('See https://arxiv.org/abs/1511.06434.pdf for more details.')
cmd:text()
cmd:text('Optional arguments:')

cmd:option('-glr', 0.0001, 'learning rate for minimizing generator loss')
cmd:option('-g_beta1', 0.5, 'value of `beta1` hyperparam for generator optimizer')
cmd:option('-dlr', 0.0001, 'learning rate for minimizing discriminator loss')
cmd:option('-d_beta1', 0.5, 'value of `beta1` hyperparam for discriminator optimizer')
cmd:option('-batch_size', 64, 'mini-batch size for training the adversarial network')
cmd:option('-num_epochs', 50, 'number of epochs to train the adversarial network')
cmd:option('-gpu', 1, 'if using GPU for training the adversarial network. Use 0 for CPU.')

opt = cmd:parse(arg)

-- create adversarial networks
local netG = DCGAN.generator(256)
local netD = DCGAN.discriminator()

-- sanity check
assert(netG.__typename == 'nn.Sequential', 'no generator network found')
assert(netD.__typename == 'nn.Sequential', 'no discriminator network found')

-- summary of models
print('----------')
print('Generator:')
print('----------')
print(netG:__tostring())
print('--------------')
print('Discriminator:')
print('--------------')
print(netD:__tostring())

-- define loss (binary cross entropy)
local criterion = nn.BCECriterion()

-- define hyperparams for adam
optimStateG = {
    learningRate=opt.glr,
    beta1=opt.g_beta1
}
optimStateD = {
    learningRate=opt.dlr,
    beta1=opt.d_beta1
}

-- move everything to the GPU
if opt.gpu > 0 then
    cutorch.setDevice(opt.gpu)
    netG:cuda()
    netD:cuda()
    criterion:cuda()
end

local parametersG, gradParametersG = netG:getParameters()
local parametersD, gradParametersD = netD:getParameters()

-- sanity check dims
input = torch.rand(2, 256):cuda()
output = netG:forward(input)
assert(torch.all(torch.eq(torch.Tensor(3, 36, 36):zero(), torch.Tensor(output[1]:size()):zero())))
prob = netD:forward(output)
assert(torch.all(torch.eq(torch.Tensor(1):zero(), torch.Tensor(prob[1]:size()):zero())))

-- training model