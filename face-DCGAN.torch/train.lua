--[[ Trains the Deep Convolutional Generative Network (DCGAN). 

Please see https://arxiv.org/abs/1511.06434.pdf 

Usage: th train.lua --help ]]

require 'torch'
require 'nn'
require 'optim'

-- local modules
local Utils = paths.dofile('utils.lua')
local Model = paths.dofile('model.lua')

local CKPT_DIR = 'ckpts'

-- command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text('Trains a Deep Convolutional Generative Network (DCGAN)')
cmd:text()
cmd:text('See https://arxiv.org/abs/1511.06434.pdf for more details.')
cmd:text()
cmd:text('Optional arguments:')

cmd:option('-z_dim', 256, 'dimensions of 1-D noise tensor to feed the generator')
cmd:option('-glr', 0.001, 'learning rate for minimizing generator loss')
cmd:option('-g_beta1', 0.5, 'value of `beta1` hyperparam for generator optimizer')
cmd:option('-dlr', 0.001, 'learning rate for minimizing discriminator loss')
cmd:option('-d_beta1', 0.5, 'value of `beta1` hyperparam for discriminator optimizer')
cmd:option('-batch_size', 64, 'mini-batch size for training the adversarial network')
cmd:option('-num_epochs', 50, 'number of epochs to train the adversarial network')
cmd:option('-gpu', 1, 'if using GPU for training the adversarial network. Use 0 for CPU.')
cmd:option('-name', 'dcgan', 'name of the adversarial networks')

opt = cmd:parse(arg)

if not (path.exists(CKPT_DIR)) then
    lfs.mkdir(CKPT_DIR)
end

-- create adversarial networks
local netG = Model.generator(opt.z_dim)
local netD = Model.discriminator()

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

-- get training data
trainset = Utils.load_dataset()

-- using GPU for training
if opt.gpu > 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    torch.setdefaulttensortype('torch.CudaTensor')
    if pcall(require, 'cudnn') then
        require 'cudnn'
        cudnn.benchmark = true
        cudnn.convert(netG, cudnn)
        cudnn.convert(netD, cudnn)
    end
    netG:cuda()
    netD:cuda()
    criterion:cuda()
    trainset:cuda()
end

-- sanity check dims
test_input = torch.rand(2, opt.z_dim)
output = netG:forward(test_input)
assert(torch.all(torch.eq(torch.Tensor(3, 36, 36):zero(), torch.Tensor(output[1]:size()):zero())),
        'generator output dims don\'t match the required spec')
prob = netD:forward(output)
assert(torch.eq(torch.Tensor(1):zero(), torch.Tensor(prob[1]:size()):zero()),
        'discriminator output dims don\'t match the required spec')
collectgarbage()

-- define hyperparams for adam
optimStateG = {
    learningRate=opt.glr,
    beta1=opt.g_beta1
}
optimStateD = {
    learningRate=opt.dlr,
    beta1=opt.d_beta1
}

-- flatten model parameters
local parametersG, gradParametersG = netG:getParameters()
local parametersD, gradParametersD = netD:getParameters()

-- training model
for epoch = 1, opt.num_epochs do
    local g_err = 0
    local d_err = 0
    local n_iter = 0
    for t = 1, trainset:size(1), opt.batch_size do
        -- disp progress bar
        xlua.progress(t, trainset:size(1))

        -- create mini-batch
        local real_images
        if opt.gpu > 0 then
            real_images = trainset[{{t, math.min(t + opt.batch_size - 1, trainset:size(1))}}]:cuda()
        else
            real_images = trainset[{{t, math.min(t + opt.batch_size - 1, trainset:size(1))}}]
        end
        local targets_real = torch.ones(real_images:size(1))
        local targets_fake = torch.zeros(real_images:size(1))

        -- create closure to evaluate f(X) and df/dX of discriminator
        local fDx = function(x)
            gradParametersD:zero()
        
            -- train with real
            local output = netD:forward(real_images)                    -- network forward
            local errD_real = criterion:forward(output, targets_real)   -- loss forward
            local df_do = criterion:backward(output, targets_real)      -- loss backward
            netD:backward(real_images, df_do)                           -- network backward
        
            -- train with fake
            local input_noise = Utils.sample_noise_batch(real_images:size(1), opt.z_dim)
            local generated = netG:forward(input_noise)
            local output = netD:forward(generated)
            local errD_fake = criterion:forward(output, targets_fake)
            local df_do = criterion:backward(output, targets_fake)
            netD:backward(generated, df_do)
        
            errD = errD_real + errD_fake
            d_err = d_err + errD
            return errD, gradParametersD
        end

        -- create closure to evaluate f(X) and df/dX of generator
        local fGx = function(x)
            gradParametersG:zero()

            -- train to make fake seem like real
            local input_noise = Utils.sample_noise_batch(real_images:size(1), opt.z_dim)
            local generated = netG:forward(input_noise)
            local output = netD:forward(generated)
            local errG = criterion:forward(output, targets_real)
            local df_do = criterion:backward(output, targets_real)
            local df_dg = netD:updateGradInput(generated, df_do)
            netG:backward(input_noise, df_dg)

            g_err = g_err + errG        
            return errG, gradParametersG
        end

        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        optim.adam(fDx, parametersD, optimStateD)
        -- (2) Update G network: maximize log(D(G(z))) -- twice so that d_loss does'nt go to 0
        for i = 1, 2 do
            optim.adam(fGx, parametersG, optimStateG)
        end
        n_iter = n_iter + 1
    end
    -- log epoch info
    print('\nEpoch[' .. epoch .. '/' .. opt.num_epochs .. ']: g_error - ' .. (g_err / n_iter) .. '  d_error - ' .. (d_err / n_iter) .. '\n')
    -- nil to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    parametersD, gradParametersD = nil, nil
    -- save checkpoints
    torch.save(path.join(CKPT_DIR, opt.name .. '_' .. epoch .. '_net_G.t7'), netG:clearState())
    torch.save(path.join(CKPT_DIR, opt.name .. '_' .. epoch .. '_net_D.t7'), netD:clearState())
    -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
    parametersD, gradParametersD = netD:getParameters()
end