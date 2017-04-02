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
cmd:option('-val_batch_info', 0)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 1e-4)
cmd:option('-grad_clip', 10)
-- For Adam people don't usually decay the learning rate
cmd:option('-lr_decay_every', 20)  -- Decay every n epochs
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-gpu', 1)

-- Train options
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
local model = nn.ConvNet(opt_clone)
model:createModel()
model:updateType(dtype)

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
local params, grad_params = train_cls.params, train_cls.grad_params

function run_on_val_data()
  model:evaluate()

  local val_data_co = coroutine.create(data_loader.next_val_batch)
  train_cls.validate(val_data_co)
  collectgarbage()
end

local optim_config = {learningRate = opt.learning_rate}
local curr_batches_processed = 0
local total_train_batches = math.floor(data_loader:train_data_size()/opt.batch_size)

for i=1, opt.max_epochs do
  train_data_co = coroutine.create(data_loader.next_train_batch)
  curr_batches_processed = 0

  while coroutine.status(train_data_co) ~= 'dead' do

    if curr_batches_processed < total_train_batches then
      local loss = train_cls.train(train_data_co, optim_config, {
        curr_epoch=i,
        total_epoch=opt.max_epochs,
        curr_batch=curr_batches_processed,
        total_batch=total_train_batches
      })

       if (opt.print_every > 0 and
            curr_batches_processed > 0 and
            curr_batches_processed % opt.print_every == 0) then
          local float_epoch = i
          local msg = 'Epoch %.2f, total epochs:%d, loss = %f'
          local args = {msg, float_epoch, opt.max_epochs, loss[1]}
          print(string.format(unpack(args)))
          print('Gradient weights for the last batch')
        end

        if (opt.validate_every_batches > 0 and 
          curr_batches_processed > 0 and
          curr_batches_processed % opt.validate_every_batches == 0) then
          run_on_val_data()
        end

        curr_batches_processed = curr_batches_processed + 1
      else
        local success, x = coroutine.resume(train_data_co, data_loader)
        assert(coroutine.status(train_data_co) == 'dead')
      end

      -- Epoch done
      -- TODO(Mohit): Maybe have a post processing step for printing etc.
 
      -- Decay learning rate
      if i % opt.lr_decay_every == 0 then
        local old_lr = optim_config.learningRate
        optim_config = {learningRate = old_lr * opt.lr_decay_factor}
      end
    end

  -- Save a checkpoint
  local check_every = opt.checkpoint_every
  if (check_every > 0 and i % check_every == 0) then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    model:evaluate()

    -- Set the dataloader to load validation data
    local val_data_co = coroutine.create(DataLoader.next_val_batch)
    train_cls.validate(val_data_co)

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      desc = opt.desc,
      epoch = i
    }
    -- Add checkpoint items
    local train_cp = train_cls.getCheckpoint()
    for k,v in pairs(train_cp) do checkpoint[k] = v end

    local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
    -- Make sure the output directory exists before we try to write it
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)

    print("DID SAVE ====>  "..filename)

    local grads_filename = string.format('%s/model_grads.json', opt.save)
    utils.write_json(grads_filename, train_cp.grads_history)


    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
    model:clearState()
    model:float()
    checkpoint.model = model
    local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
    paths.mkdir(paths.dirname(filename))
    torch.save(filename, checkpoint)
    model:type(dtype)
    train_cls.params, train_cls.grad_params = model:getParameters()
    params, grad_params = train_cls.params, train_cls.grad_params
    collectgarbage()

  end
    
end

