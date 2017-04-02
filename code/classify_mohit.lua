require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'

require 'util.DataLoader'
require 'model.ConvNet'

local utils = require 'util.utils'

local dtype = 'torch.FloatTensor'

local cmd = torch.CmdLine()

cmd:option('-desc', '')
-- Dataset options
cmd:option('-train_h5', '')
cmd:option('-test_h5', '')
cmd:option('-num_classify', 10)
cmd:option('-batch_size', 100)
cmd:option('-val_batch_info', 1)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 1e-4)
cmd:option('-grad_clip', 10)
-- For Adam people don't usually decay the learning rate
cmd:option('-lr_decay_every', 20)  -- Decay every n epochs
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-gpu', 1)

-- Train options
cmd:option('-init_from', '')
cmd:option('-use_entropy_reg', 1)
cmd:option('-coef_beta_const', 0.8)
cmd:option('-coef_beta_reg', 1)
cmd:option('-coef_beta_start', 1)
cmd:option('-coef_beta_end', 0.1) 
cmd:option('-coef_beta_decay_steps', -1) -- If -1 use total # of iters
cmd:option('-train_layer', 'train_bootstrap_var')
cmd:option('-target_loss_coef', 1)
cmd:option('-pred_loss_coef', 1)
cmd:option('-beta_loss_coef', 0.1)
cmd:option('-beta_reg_loss_coef', 0.01)

-- Output options
cmd:option('-save', '')
cmd:option('-print_every', 200)-- Print every n batches
cmd:option('-checkpoint_every', 1)  -- Checkpoint after every n epochs
cmd:option('-checkpoint_name', 'checkpoint')
cmd:option('-validate_every_batches', 75) -- Run on validation data ever n batches
cmd:option('-train_log', 'train.log')
cmd:option('-test_log', 'test.log')
cmd:option('-test_batch', 'test_batch.json')
cmd:option('-test_scores', 'test_scores.json')
cmd:option('-test_preds', 'test_preds.json')
cmd:option('-test_beta', 'test_beta.json')
cmd:option('-save_test_data_stats', 1)
cmd:option('-debug_weights', 1)

torch.manualSeed(1)

local opt = cmd:parse(arg)
opt.checkpoint_name = opt.save..'/'..opt.checkpoint_name
opt.test_scores = opt.save .. '/' .. opt.test_scores
opt.test_preds = opt.save .. '/' .. opt.test_preds
opt.test_batch = opt.save .. '/' .. opt.test_batch
opt.test_beta = opt.save .. '/' .. opt.test_beta

if opt.gpu == 1 then
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.setDevice(1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU 1'))
else
  print('Running in CPU mode')
end

local classes = {}
for i=1,opt.num_classify do table.insert(classes, i) end
local train_conf = optim.ConfusionMatrix(classes)
local val_conf = optim.ConfusionMatrix(classes)

local opt_clone = torch.deserialize(torch.serialize(opt))
-- GLobal options variable is a global
G_global_opts = opt_clone
print(G_global_opts)


-- Get the main training objects
local data_loader = DataLoader(opt_clone)
local model = torch.load(opt_clone.init_from)
model = model.model
model = model:updateType(G_global_opts.dtype)

local train_cls
if opt.train_layer == 'train_simple' then
  train_cls = require 'train_layer/train_simple'
elseif opt.train_layer == 'train_bootstrap_var' then
  train_cls = require 'train_layer/train_bootstrap_var'
elseif opt.train_layer == 'train_bootstrap_const' then
  train_cls = require 'train_layer/train_bootstrap_const'
else
  assert(false)
end

train_cls.setup{
  dtype=dtype,
  model=model,
  train_conf=train_conf,
  val_conf=val_conf,
  data_loader=data_loader,
}

data_co = coroutine.create(DataLoader.next_val_batch)
-- second argument is save_test_data_stats=1
train_cls.validate(DataLoader.next_val_batch, 1)
local test_data_stats = train_cls.test_data_stats

print(val_conf)
-- Save the test batch data
paths.mkdir(paths.dirname(opt.test_batch))
utils.write_json(opt.test_batch, test_data_stats.test_data)

-- Save the scores
paths.mkdir(paths.dirname(opt.test_scores))
utils.write_json(opt.test_scores, test_data_stats.test_scores)

-- Save the predictions
paths.mkdir(paths.dirname(opt.test_preds))
utils.write_json(opt.test_preds, test_data_stats.test_preds)

-- Save betas
paths.mkdir(paths.dirname(opt.test_beta))
utils.write_json(opt.test_beta, test_data_stats.test_beta)

