--[[ Utility modules for training DCGAN ]]--

local Utils = {}
Utils.__index = Utils

function Utils.sample_noise_batch(batch_size, z_dim)
    return torch.Tensor(batch_size, z_dim):normal(0.0, 1.0)
end

function Utils.load_dataset()
    
end