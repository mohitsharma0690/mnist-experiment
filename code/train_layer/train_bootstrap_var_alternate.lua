local utils = require '../util/utils.lua'

require 'nn'
require 'cunn'
require 'hdf5'
require 'math'
require 'paths'
require 'optim'

require 'model.MS_BootstrapCrossEntropy'
require 'model.MS_EntropyRegLoss'

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
    -- This is just beta*crit1 + (1-beta)*crit2
    pred_loss_history={}, 
    beta_loss_history={},
    beta_reg_loss_history={},
    val_loss_history={},
    beta_history={},
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

  -- The last layer from the model would be 128x5. We remove it to add a
  -- parallel table that predicts y_hat and Beta
  if G_global_opts.save_test_data_stats ~= 1 then
    self.model.net:remove(10) -- Remove last linear (class layer)

    self.model.net:add(nn.Replicate(2)) -- Copy into two tensors
    self.model.net:add(nn.SplitTable(1)) -- Split above tensor into two outputs
    self.final_table = nn.ParallelTable()

    -- Add the prediction model
    self.pred_model = nn.Sequential()
    self.pred_model:add(nn.Linear(200, 10))
    self.pred_model:add(nn.LogSoftMax())
    self.final_table:add(self.pred_model)

    -- Compute Beta
    self.beta_prob = nn.Sequential()
    self.beta_prob:add(nn.Linear(200, 1))
    self.beta_prob:add(nn.Sigmoid())
    self.final_table:add(self.beta_prob)

    self.model.net:add(self.final_table)

    self.model:updateType(self.dtype)
  end


  print(self.model.net)
  
  self.crit1 = nn.ClassNLLCriterion():type(self.dtype)
  self.crit2 = nn.ClassNLLCriterion():type(self.dtype)

  self.beta_crit = nn.MS_BootstrapCrossEntropy():type(self.dtype)
  if G_global_opts.use_entropy_reg == 1 then
    self.beta_reg = nn.MS_EntropyRegLoss(3.0):type(self.dtype)
  else
    self.beta_reg = nn.MSECriterion():type(self.dtype)
  end
  self.coef_beta = G_global_opts['coef_beta_reg']
  self.coef_beta_start = self.coef_beta
  self.coef_beta_end = 0.5
  assert(self.coef_beta ~= nil)
  
  self.target_loss_coef = utils.get_kwarg(G_global_opts, 'target_loss_coef')
  self.pred_loss_coef = utils.get_kwarg(G_global_opts, 'pred_loss_coef')
  self.beta_loss_coef = utils.get_kwarg(G_global_opts, 'beta_loss_coef')
  self.beta_reg_loss_coef = utils.get_kwarg(G_global_opts, 'beta_reg_loss_coef')
  
  self.params, self.grad_params = self.model:getParameters()
end

function train_cls.get_current_beta(stats)
  local self = train_cls
  local total_it = stats.total_epoch * stats.total_batch
  local done_it = stats.curr_epoch * stats.total_batch
  local step = (self.coef_beta_start - self.coef_beta_end) / total_it
  local curr_beta = self.coef_beta_start - done_it * step
  if curr_beta < self.coef_beta_end then curr_beta = self.coef_beta_end end
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

function train_cls.f(w)
  local self = train_cls
  assert(w == self.params)
  self.grad_params:zero()

  local success, x, y

  -- Get the data and store it for the next iteration (to optimize Beta)
  if self.train_pred_model then
    success, x, y = self.read_data_co(self.data_co, self.data_loader)
    if not x then return 0, self.grad_params end
    if x ~= nil then 
      self.train_x = torch.deserialize(torch.serialize(x))
      self.train_y = y:clone()
    else
      self.train_x, self.train_y = nil, nil
    end
  else
    success, x, y = true, self.train_x, self.train_y
    if x == nil or y ==  nil then success = false end
  end

  x = utils.convert_to_type(x, self.dtype)
  y = utils.convert_to_type(y, self.dtype)

  local scores = self.model:forward(x)  -- scores is a table
  -- scores[1] is the target scores and scores[2] are the Beta scores (confidence)
  assert(torch.max(scores[1]) == torch.max(scores[1]))
  local _, preds = torch.max(scores[1], 2)
  local beta = scores[2]:clone()

  -- We want Beta to be 1 (Get Beta and (1-Beta) tensors)
  local expected_beta = torch.ones(beta:size())
  expected_beta = expected_beta:type(self.dtype)
  local one_minus_beta = torch.add(torch.ones(beta:size()):cuda(), -1, beta)

  scores = scores[1]:clone()

  local loss  -- This is the total loss

  if self.train_pred_model then
    -- This is sum(y*log(y_hat))
    local loss_target = self.target_loss_coef * self.crit1:forward(scores, y)
    -- This is sum(y_hat * log(y_hat))
    local preds_vec = preds:view(-1)
    local loss_pred = self.pred_loss_coef * self.crit2:forward(scores, preds_vec)

    -- Get gradients for the above two criterions
    local grad_target = self.crit1:backward(scores, y):mul(self.target_loss_coef)
    local grad_pred = self.crit2:backward(scores, preds_vec):mul(self.pred_loss_coef)

    -- Expand beta across rows since we use the same value across
    -- all classes
    local beta_exp = torch.expand(beta, beta:size(1), grad_target:size(2))
    local one_minus_beta_exp = torch.expand(one_minus_beta, beta:size(1), grad_pred:size(2))
    -- Get the cumulative cross entropy gradients by linear combination
    local grad_scores = torch.add(
        grad_target:cmul(beta_exp), grad_pred:cmul(one_minus_beta_exp))

    local final_grad_scores = {grad_scores, torch.Tensor(beta:size()):zero()}
    final_grad_scores[2] = final_grad_scores[2]:type(self.dtype)

    -- Backprop the gradient
    self.model:backward(x, final_grad_scores)

    -- Update confusion matrix
    self.train_conf:batchAdd(scores, y)

    -- This by itself is not a correct estimation of the loss but for now
    -- its Ok. Since we should do beta*loss_target + (1-beta)*loss_pred
    local total_pred_loss = torch.add(
        loss_target * beta, loss_pred * one_minus_beta)
    total_pred_loss = total_pred_loss:sum() / y:size(1)

    loss = total_pred_loss

    -- Add to history
    table.insert(self.checkpoint.crit1_loss_history, loss_target)
    table.insert(self.checkpoint.crit2_loss_history, loss_pred)
    table.insert(self.checkpoint.pred_loss_history, total_pred_loss)

  else

    -- Calculate the loss with Beta as the target
    local loss_beta = self.beta_loss_coef * self.beta_crit:forward(beta, {y, scores})

    -- Use custom (annealed) beta regularization
    local loss_reg = self.curr_coef_beta * self.beta_reg:forward(beta, expected_beta)
    loss_reg = self.curr_coef_beta * self.beta_reg_loss_coef * loss_reg

    local grad_beta = self.beta_crit:backward(beta, {y, scores}):mul(self.beta_loss_coef)
    local grad_reg = self.beta_reg:backward(beta, expected_beta):mul(self.curr_coef_beta*self.beta_reg_loss_coef)

    grad_beta = grad_beta:add(grad_reg)

    local final_grad_scores = {torch.Tensor(scores:size()):zero(), grad_beta}
    final_grad_scores[1] = final_grad_scores[1]:type(self.dtype)

    -- Backprop the gradient
    self.model:backward(x, final_grad_scores)

    loss = loss_beta + loss_reg

    -- Add to history
    table.insert(self.checkpoint.beta_loss_history, loss_beta)
    table.insert(self.checkpoint.beta_reg_loss_history, loss_reg)
    table.insert(self.checkpoint.beta_history, self.curr_coef_beta)
  end

  if G_global_opts.grad_clip > 0 then
    self.grad_params:clamp(
        -G_global_opts.grad_clip, G_global_opts.grad_clip)
  end

  if G_global_opts.debug_weights == 1 then 
    local curr_grad_history = self.model:getGradWeights(loss, x, y) 
    table.insert(self.checkpoint.grads_history, curr_grad_history)
  end

  return loss, self.grad_params
end

function train_cls.validate(val_data_co, save_stats)
  local self = train_cls
  local val_loss, num_val = 0, 0
  local save = save_stats or 0

  self.val_conf:zero()

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv, batch = coroutine.resume(
        val_data_co, self.data_loader) 

    if success and xv ~= nil then
      xv = utils.convert_to_type(xv, self.dtype)
      yv = utils.convert_to_type(yv, self.dtype)

      local scores = self.model:forward(xv)
      assert(torch.max(scores[1]) == torch.max(scores[1]))
      val_loss = val_loss + self.crit1:forward(scores[1], yv)

      -- Since its LogSoftMax we convert it into Softmax to avoid
      -- NaN issues
      scores[1] = torch.exp(scores[1])
      self.val_conf:batchAdd(scores[1], yv)
      num_val = num_val + 1

      if save == 1 then
        local logs = self.test_data_stats
        local scores_max, scores_max_idx = torch.max(scores[1], 2)
        for i=1,#batch do table.insert(logs.test_data, batch[i]) end
        scores_max_idx = torch.totable(scores_max_idx)
        for i=1,#scores_max_idx do table.insert(logs.test_preds, scores_max_idx[i]) end
        local scores_table = torch.totable(scores[1])
        for i=1,#scores_table do table.insert(logs.test_scores, scores_table[i]) end
        local beta = torch.totable(scores[2])
        for i=1,#beta do table.insert(logs.test_beta, beta[i]) end
      end

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

  self.curr_coef_beta = self.get_current_beta(stats)
  self.train_pred_model = true

  local loss1, loss2
  _, loss1 = optim.adam(self.f, self.params, optim_config)
  table.insert(self.checkpoint.train_loss_history, loss1[1]) 

  self.train_pred_model = false
  _, loss2 = optim.adam(self.f, self.params, optim_config)

  local msg = 'Epoch: [%d/%d]\t Iteration:[%d/%d]\tTarget(y) loss: %.2f\t'
  msg = msg..'y_hat loss: %.2f\t Actual pred loss: %.2f\t'
  msg = msg..'beta_loss: %.2f\t beta_reg_loss: %.2f\t beta:%.2f'

  local logs = self.checkpoint
  local args = {
    msg,
    stats.curr_epoch, stats.total_epoch,
    stats.curr_batch, stats.total_batch,
    logs.crit1_loss_history[#logs.crit1_loss_history],
    logs.crit2_loss_history[#logs.crit2_loss_history],
    logs.pred_loss_history[#logs.pred_loss_history],
    logs.beta_loss_history[#logs.beta_loss_history],
    logs.beta_reg_loss_history[#logs.beta_reg_loss_history],
    logs.beta_history[#logs.beta_history],
  }
  print(string.format(unpack(args)))

  return loss1
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
