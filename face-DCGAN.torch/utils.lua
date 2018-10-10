--[[ Utility modules for training DCGAN ]]--

require 'image'
require 'torch'

local Utils = {}
Utils.__index = Utils

local PREPRO_DIR = 'data/preprocessed_images'

function Utils.sample_noise_batch(batch_size, z_dim)
    return torch.Tensor(batch_size, z_dim):normal(0.0, 1.0)
end

function Utils.load_dataset()
    local images = {}
    local i = 0
    for file in paths.files(PREPRO_DIR, function(nm) return nm:find('.jpg') end) do
        i = i + 1
        local img = image.load(path.join(PREPRO_DIR, file))
        table.insert(images, img)
        if channels == nil then
            channels = img:size(1)
            height = img:size(2)
            width = img:size(3)
        end
    end
    local data = torch.Tensor(i, channels, height, width)
    for i = 1, #images do
        data[i] = images[i]
        images[i] = nil
    end
    -- preprocess images
    data:div(127.5)
    data:add(-1.0)
    return data
end

return Utils