local utils = require '../util/utils.lua'

require 'nn'
require 'cunn'
require 'hdf5'
require 'math'
require 'paths'
require 'optim'

require 'model.MS_BootstrapCrossEntropy'

--[[
-- In this method we are using the top2 predicted labels from y_hat as our 
-- pseudo-label. We have to use binary cross entropy (BCECriterion) since NLL
-- assumes targets to be in one-hot form.
]]

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

  -- Since we are using temperature (here T=2)
  self.model.net:add(nn.MulConstant(1.0/2))
  -- Since we will be using NLL criterion
  self.model.net:add(nn.SoftMax())
  self.model:updateType(self.dtype)
  print(self.model.net)
  
  self.crit1 = nn.BCECriterion():type(self.dtype)
  -- self.crit2 = nn.ClassNLLCriterion():type(self.dtype)
  --
   
  self.target_loss_coef = utils.get_kwarg(G_global_opts, 'target_loss_coef')
  self.pred_loss_coef = utils.get_kwarg(G_global_opts, 'pred_loss_coef')
  self.beta_loss_coef = utils.get_kwarg(G_global_opts, 'beta_loss_coef')
  self.beta_reg_loss_coef = utils.get_kwarg(G_global_opts, 'beta_reg_loss_coef')

  self.params, self.grad_params = self.model:getParameters()
end

function train_cls.get_current_beta(stats)
  -- Finish
  local self = train_cls
  local total_it = stats.total_epoch * stats.total_batch
  local decay_it = G_global_opts.coef_beta_decay_steps
  if decay_it == -1 then decay_it = total_it end
  local done_it = (stats.curr_epoch - 1) * stats.total_batch + stats.curr_batch

  if done_it < 2000 then return 0.999 end

  local step = (self.coef_beta_start - self.coef_beta_end) / decay_it
  local curr_beta = self.coef_beta_start - done_it * step
  if curr_beta < self.coef_beta_end then curr_beta = self.coef_beta_end end

  -- No beta regularization after 5 epochs!!
  --if stats.curr_epoch > 5 then return 0 end
  return curr_beta
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
  local one_hot_y = utils.get_one_hot_tensor(y, G_global_opts.num_classify)
  one_hot_y = utils.convert_to_type(one_hot_y, self.dtype)

  local scores = self.model:forward(x) 
  -- Get the top 2 (largest) values in sorted order 
  local _, topk_preds = torch.topk(scores, 2, true, true)
  -- Get the pseudo labels 
  local pseudo_label = torch.Tensor(scores:size()):zero()
  for i=1, topk_preds:size(1) do 
    pseudo_label[i][topk_preds[i][1]] = 1
    --if y[i] ~= topk_preds[i][1] then
    --  pseudo_label[i][topk_preds[i][1]] = 1
    --else
    --  pseudo_label[i][topk_preds[i][2]] = 1 
    --end
  end
  pseudo_label = utils.convert_to_type(pseudo_label, self.dtype)

  -- This is sum(y*log(y_hat))
  local new_target = torch.add(self.beta*one_hot_y, (1-self.beta)*pseudo_label)
  local loss = self.crit1:forward(scores, new_target)

  local grad_scores = self.crit1:backward(scores, new_target)

  -- Backrop the gradient
  self.model:backward(x, grad_scores)

  -- Update the confusion matrix
  self.train_conf:batchAdd(scores, y)

  table.insert(self.checkpoint.crit1_loss_history, loss)
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
      local one_hot_yv = utils.get_one_hot_tensor(yv, G_global_opts.num_classify)
      one_hot_yv = utils.convert_to_type(one_hot_yv, self.dtype)

      local scores = self.model:forward(xv)
      val_loss = val_loss + self.crit1:forward(scores, one_hot_yv)

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

  if (stats.curr_batch > 0 and stats.curr_batch % G_global_opts.print_every == 0) then 

    local msg = 'Epoch: [%d/%d]\t Iteration:[%d/%d]\tTarget(y) loss: %.2f'

    local logs = self.checkpoint
    local args = {
      msg,
      stats.curr_epoch, stats.total_epoch,
      stats.curr_batch, stats.total_batch,
      logs.train_loss_history[#logs.train_loss_history],
    }
    print(string.format(unpack(args)))
    print("Gradients: ")
    print(logs.grads_history[#logs.grads_history]) 
  end

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

