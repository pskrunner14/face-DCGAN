--[[ Trains the Deep Convolutional Generative Network (DCGAN). 

Please see https://arxiv.org/abs/1511.06434.pdf 

Usage: th train.lua --help ]]

require 'torch'
require 'nn'
require 'optim'
require 'image'

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
cmd:option('-dlr', 0.001, 'learning rate for minimizing discriminator loss')
cmd:option('-batch_size', 64, 'mini-batch size for training the adversarial network')
cmd:option('-num_epochs', 50, 'number of epochs to train the adversarial network')
cmd:option('-save_every', 5, 'epoch interval to save model checkpoints')
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
    learningRate=opt.glr
}
optimStateD = {
    learningRate=opt.dlr
}

-- flatten model parameters
local parametersG, gradParametersG = netG:getParameters()
local parametersD, gradParametersD = netD:getParameters()

-- define input placeholders
input = torch.Tensor(opt.batch_size, trainset[1]:size(1), 
                    trainset[1]:size(2), trainset[1]:size(3))
noise = torch.Tensor(opt.batch_size, opt.z_dim)
targets = torch.ones(opt.batch_size)

local limit_index = trainset:size(1) - (trainset:size(1) % opt.batch_size) + 1

-- training model
for epoch = 1, opt.num_epochs do
    local g_err, d_err, n_iter = 0, 0, 0
    -- loop over batches
    for t = 1, limit_index, opt.batch_size do
        -- disp progress bar
        xlua.progress(t, limit_index)
        -- break out of loop when we can't make batches of required size
        if t == limit_index then
            break
        end
        -- create mini-batch
        input:copy(trainset[{{t, t + opt.batch_size - 1}}])

        -- create closure to evaluate f(X) and df/dX of discriminator
        local fDx = function(x)
            gradParametersD:zero()
        
            targets:fill(1)
            -- train with real
            local output = netD:forward(input)                    -- network forward
            local errD_real = criterion:forward(output, targets)   -- loss forward
            local df_do = criterion:backward(output, targets)      -- loss backward
            netD:backward(input, df_do)                           -- network backward
        
            noise:normal(0.0, 1.0)
            targets:fill(0)
            -- train with fake
            local generated = netG:forward(noise)
            input:copy(generated)
            local output = netD:forward(input)
            local errD_fake = criterion:forward(output, targets)
            local df_do = criterion:backward(output, targets)
            netD:backward(input, df_do)
        
            errD = errD_real + errD_fake
            d_err = d_err + errD
            return errD, gradParametersD
        end

        -- create closure to evaluate f(X) and df/dX of generator
        local fGx = function(x)
            gradParametersG:zero()

            targets:fill(1)
            -- train to make fake seem like real
            -- use the output and generated from above to save on computation
            local output = netD.output
            local errG = criterion:forward(output, targets)
            local df_do = criterion:backward(output, targets)
            local df_dg = netD:updateGradInput(input, df_do)
            netG:backward(input, df_dg)

            g_err = g_err + errG

            -- update inputs for next iteration
            if first then
                noise:normal(0.0, 1.0)
                local generated = netG:forward(noise)
                input:copy(generated)
                netD:forward(input)
            end
            return errG, gradParametersG
        end

        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        optim.adam(fDx, parametersD, optimStateD)
        -- (2) Update G network: maximize log(D(G(z)))
        first = true
        for i = 1, 2 do
            optim.adam(fGx, parametersG, optimStateG)
            first = false
        end
        n_iter = n_iter + 1
    end
    -- log epoch info
    print('\nEpoch[' .. epoch .. '/' .. opt.num_epochs .. ']: g_error - ' .. (g_err / n_iter) .. '  d_error - ' .. (d_err / n_iter) .. '\n')
    -- nil to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    parametersD, gradParametersD = nil, nil
    -- save checkpoints
    if epoch % opt.save_every == 0 then
        print('saving checkpoints in `ckpts/`')
        torch.save(path.join(CKPT_DIR, opt.name .. '_' .. epoch .. '_net_G.t7'), netG:clearState())
        torch.save(path.join(CKPT_DIR, opt.name .. '_' .. epoch .. '_net_D.t7'), netD:clearState())
    end
    -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
    parametersD, gradParametersD = netD:getParameters()
end