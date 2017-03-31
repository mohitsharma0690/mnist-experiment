local utils = require '../util/utils.lua'

require 'nn'
require 'cunn'
require 'hdf5'
require 'math'
require 'paths'
require 'optim'

require 'model.MS_BootstrapCrossEntropy'

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
    crit1_loss_history={},
    crit2_loss_history={},
    val_loss_history={},
    grads_history={},
  }

  if G_global_opts.save_test_data_stats == 1 then
    self.test_data_stats = {
      test_scores = {},
      test_preds = {},
      test_data = {},
      test_beta = {},
    }
  end

  self.beta = G_global_opts['coef_beta_const']

  -- Since we will be using NLL criterion
  self.model.net:add(nn.LogSoftMax())
  self.model:updateType(self.dtype)
  print(self.model.net)
  
  self.crit1 = nn.ClassNLLCriterion():type(self.dtype)
  self.crit2 = nn.ClassNLLCriterion():type(self.dtype)

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

function train_cls.f_opt_together(w)
  local self = train_cls
  assert(w == self.params)
  self.grad_params:zero()

  local success, x, y = self.read_data_co(self.data_co, self.data_loader)
  if not x then return 0, self.grad_params end
  
  x = utils.convert_to_type(x, self.dtype)
  y = utils.convert_to_type(y, self.dtype)

  local scores = self.model:forward(x) 
  local _, preds = torch.max(scores, 2)
  preds = preds:view(-1)


  -- This is sum(y*log(y_hat))
  local loss_target = self.crit1:forward(scores, y)
  local loss_pred = self.crit2:forward(scores, preds)
  local loss = self.beta*loss_target + (1-self.beta)*loss_pred

  local grad_target = self.crit1:backward(scores, y)
  grad_target = grad_target:mul(self.beta)
  local grad_pred = self.crit2:backward(scores, preds)
  grad_pred = grad_pred:mul(1.0 - self.beta)

  local grad_scores = torch.add(grad_target, grad_pred)

  -- Backrop the gradient
  self.model:backward(x, grad_scores)

  -- Update the confusion matrix
  self.train_conf:batchAdd(scores, y)

  table.insert(self.checkpoint.crit1_loss_history, loss_target)
  table.insert(self.checkpoint.crit2_loss_history, loss_pred)
  table.insert(self.checkpoint.train_loss_history, loss)

  if G_global_opts.grad_clip > 0 then
    self.grad_params:clamp(-G_global_opts.grad_clip, G_global_opts.grad_clip)
  end

  if G_global_opts.debug_weights == 1 then 
    local curr_grad_history = self.model:getGradWeights(loss, x, y) 
    table.insert(self.checkpoint.grads_history, curr_grad_history)
  end

  return loss, self.grad_params
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
      val_loss = val_loss + self.crit1:forward(scores, yv)

      -- Since its LogSoftMax we convert it into Softmax to avoid
      -- NaN issues
      scores = torch.exp(scores)
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
  self.model:training()

  local loss
  _, loss = optim.adam(
      self.f_opt_together, self.params, optim_config)
  local msg = 'Epoch: [%d/%d]\t Iteration:[%d/%d]\tTarget(y) loss: %.2f\t'
  msg = msg..'Target loss: %.2f\t self pred loss: %.2f\t'

  local logs = self.checkpoint
  local args = {
    msg,
    stats.curr_epoch, stats.total_epoch,
    stats.curr_batch, stats.total_batch,
    logs.train_loss_history[#logs.train_loss_history],
    logs.crit1_loss_history[#logs.crit1_loss_history],
    logs.crit2_loss_history[#logs.crit2_loss_history],
  }
  print(string.format(unpack(args)))

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

