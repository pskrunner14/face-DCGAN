--[[ Deep Convolutional Generative Adversarial Network (DCGAN).

Using deep convolutional generative adversarial networks 
to generate face images from a noise distribution.

References:
    - Generative Adversarial Nets. Goodfellow et al. arXiv: 1406.2661.
    - Unsupervised Representation Learning with Deep Convolutional 
    Generative Adversarial Networks. A Radford, L Metz, S Chintala. 
    arXiv: 1511.06434.

Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434.pdf).

Author: Prabhsimran Singh
Project: https://github.com/pskrunner14/face-DCGAN/tree/master/face-DCGAN.torch ]]--

-- Randomly initializes the weights and biases of a network module.
local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
       m.weight:normal(0.0, 0.02)
       m.bias:zero()
    elseif name:find('BatchNormalization') then
       if m.weight then m.weight:normal(1.0, 0.02) end
       if m.bias then m.bias:zero() end
    end
 end 

-- DCGAN model
local DCGAN = {}

-- DCGAN Generator network as described by
-- Radford et al. in "Unsupervised Representation Learning 
-- with Deep Convolutional Generative Adversarial Networks".
function DCGAN.generator(z_dim)
    local net = nn.Sequential()
    --[[ DENSE 1 ]]--
    net:add(nn.Linear(z_dim, 8 * 8 * 10))
    net:add(nn.BatchNormalization(8 * 8 * 10))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.View(-1, 10, 8, 8))
    --[[ DECONV 1 ]]--
    net:add(nn.SpatialFullConvolution(10, 64, 5, 5))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.LeakyReLU(0.2, true))
    --[[ DECONV 2]]--
    net:add(nn.SpatialFullConvolution(64, 64, 5, 5))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.LeakyReLU(0.2, true))
    --[[ UPSAMPLING ]]--
    net:add(nn.SpatialUpSamplingNearest(2))
    --[[ DECONV 3 ]]--
    net:add(nn.SpatialFullConvolution(64, 32, 7, 7))
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.LeakyReLU(0.2, true))
    --[[ FINAL CONV ]]--
    net:add(nn.SpatialConvolution(32, 3, 3, 3))
    net:add(nn.Tanh())

    net:apply(weights_init)
    return net
end

-- DCGAN Discriminator network as described by
-- Radford et al. in "Unsupervised Representation Learning 
-- with Deep Convolutional Generative Adversarial Networks".
function DCGAN.discriminator()
    local net = nn.Sequential()
    --[[ CONV 1 ]]--
    net:add(nn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1)) -- same padding with stride 1
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.SpatialAveragePooling(2, 2, 1, 1))
    --[[ CONV 2 ]]--
    net:add(nn.SpatialConvolution(32, 32, 3, 3, 2, 2, 1, 1)) -- same padding with stride 2
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.LeakyReLU(0.2, true))
    net:add(nn.SpatialAveragePooling(2, 2, 1, 1))
    -- --[[ DENSE 1 ]]--
    net:add(nn.View(-1):setNumInputDims(3))
    net:add(nn.Linear(9248, 256))
    net:add(nn.BatchNormalization(256))
    net:add(nn.LeakyReLU(0.2, true))
    -- -- --[[ FINAL DENSE ]]--
    net:add(nn.Linear(256, 1))
    net:add(nn.Sigmoid())

    net:apply(weights_init)
    return net
end

return DCGAN