--[[ Preprocessing script for training DCGAN ]]--

require 'lfs'
require 'xlua'
require 'paths'
require 'image'
require 'torch'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Preprocesses images and saves them as Tensor file (.t7) for training.')
cmd:text()
cmd:text('Optional arguments:')

cmd:option('-x1', 80, '')
cmd:option('-y1', 80, '')
cmd:option('-prescale_x', 90, '')
cmd:option('-prescale_y', 90, '')
cmd:option('-width', 36, '')
cmd:option('-height', 36, '')
cmd:option('-num_images', 13233, '')

opt = cmd:parse(arg)

local DATASET_DIR = '../datasets/lfw-deepfunneled'
local DATA_DIR = 'data'
local PREPRO_DIR = 'preprocessed_images'
local out_tensor_file = 'train_data.t7'

if not (path.exists(path.join(DATA_DIR, PREPRO_DIR))) then
    if not (path.exists(DATA_DIR)) then
        lfs.mkdir(DATA_DIR)
    end
    lfs.mkdir(path.join(DATA_DIR, PREPRO_DIR))
end

if not(path.exists(path.join(DATA_DIR, out_tensor_file))) then
    local data = {}
    local iter = 0
    print('Preprocessing images:')
    for folder in paths.files(DATASET_DIR) do
        for file in paths.files(path.join(DATASET_DIR, folder), function(nm) return nm:find('.jpg') end) do
            xlua.progress(iter, opt.num_images)
            local img
            if not (path.exists(path.join(DATA_DIR, PREPRO_DIR, file))) then
                img = image.load(path.join(DATASET_DIR, folder, file))
                img = image.crop(img, opt.x1, opt.y1, opt.x1 + opt.prescale_x, opt.y1 + opt.prescale_y)
                img = image.scale(img, opt.width, opt.height)
                image.save(path.join(DATA_DIR, PREPRO_DIR, file), img)
            else
                img = image.load(path.join(DATA_DIR, PREPRO_DIR, file))
            end
            table.insert(data, img)
            iter = iter + 1
        end
    end
    print('\nSaving preprocessed data into `' .. out_tensor_file .. '`')
    torch.save(path.join(DATA_DIR, out_tensor_file), data)
end