local utils = require '../util/utils.lua'

require 'nn'
require 'cunn'
require 'hdf5'
require 'math'
require 'paths'
require 'optim'

require 'model.ConvNet'

local train_cls = {}

function train_cls.setup(args)
  local self = train_cls
  self.dtype = args.dtype
  self.data_loader = args.data_loader
  self.model = args.model
  self.train_conf = args.train_conf
  self.val_conf = args.val_conf
  self.checkpoint = {
    train_loss_history={},
    val_loss_history={},
    grads_history={},
  }
  
  self.params, self.grad_params = self.model:getParameters()
end

function train_cls.read_data_co(data_co, data_loader)
  local success, x, y = coroutine.resume(data_co, data_loader)
  if not success then print('Data couroutine returns fail.') end
  if x == nil then
    print('x is nil returning 0 loss')
  elseif success == false then
    -- print crash logs
    if torch.isTensor(x) then 
      print(x:size())
    else
      print(x)
    end
  end
  return success,x,y
end

function train_cls.f(w)
  local self = train_cls
  assert(w == self.params)
  self.grad_params:zero() 

  local success, x, y = self.read_data_co(self.data_co, self.data_loader)
  if not x then return 0, self.grad_params end

  x = utils.convert_to_type(x, self.dtype)
  y = utils.convert_to_type(y, self.dtype)

  local scores = self.model:forward(x)  -- scores is a table
  local loss = crit:forward(scores, y)
  local grad_scores = crit:backward(scores, y) 
  model:backward(x, grad_scores)

  if G_global_opts.grad_clip > 0 then
    grad_params:clamp(-G_global_opts.grad_clip, G_global_opts.grad_clip)
  end

  -- Add to confusion matrix
  self.train_conf:batchAdd(scores, y)

  -- Add losses to history
  table.insert(self.checkpoint.train_loss_history, loss)

  if G_global_opts.debug_weights == 1 then
    local curr_grad_history = self.model:getGradWeights(loss, x, y) 
    table.insert(self.checkpoint.grads_history, curr_grad_history)
  end

  return loss, grad_params
end

function train_cls.validate(val_data_co)
  local self = train_cls
  local val_loss, num_val = 0, 0

  self.val_conf:zero()

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv = coroutine.resume(
        val_data_co, self.data_loader) 

    if success and xv ~= nil then
      xv = utils.convert_to_type(xv, self.dtype)
      yv = utils.convert_to_type(yv, self.dtype)

      local scores = self.model:forward(xv)
      val_loss = val_loss + self.crit:forward(scores, yv)

      self.val_conf:batchAdd(scores, yv)
      num_val = num_val + 1
    elseif success ~= true then
      print('Validation data coroutine failed')
      print(xv)
    end
  end

  val_loss = val_loss / num_val
  print('val_loss = ', val_loss)
  table.insert(self.checkpoint.val_loss_history, val_loss)
  print(self.val_conf)
  self.model:training()
end

function train_cls.train(train_data_co, optim_config, stats)
  local self = train_cls
  self.data_co = train_data_co
  model:training()

  local loss
  _, loss = optim.adam(self.f, self.params, optim_config)
  print(string.format('Train loss: %.2f', loss[1]))

  return loss
end

function train_cls.getCheckpoint()
  local self = train_cls
  local cp = {}
  for k,v in pairs(self.checkpoint) do cp[k] = v end
  cp.train_conf = self.train_conf:__tostring__()
  cp.val_conf = self.val_conf:__tostring__()
  return cp
end

return train_cls

