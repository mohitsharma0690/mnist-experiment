local cjson = require 'cjson'

require 'hdf5' 
require 'math'
require 'paths'

local utils = {}

function utils.get_kwarg(kwargs, name, default_val)
  if kwargs == nil then return default_val end
  if kwargs[name] == nil and default_val == nil then
    assert(false, string.format('"%s" expected and not given', name))
  elseif kwargs[name] == nil then
    return default_val
  else
      return kwargs[name]
  end
end

function utils.read_json(path)
  local f = io.open(path, 'r')
  local s = f:read('*all')
  f:close()
  return cjson.decode(s)
end


function utils.write_json(path, obj)
  local s = cjson.encode(obj)
  local f = io.open(path, 'w')
  f:write(s)
  f:close()
end

function utils.convert_to_type(x, dtype)
  if torch.isTensor(x) then 
    x = x:type(dtype)
    return x
  end
  for i=1,#x do 
    if torch.isTensor(x[i]) then x[i] = x[i]:type(dtype)
    else for j=1,#x[i] do x[i][j] = x[i][j]:type(dtype) end end
  end
  return x
end

function utils.get_one_hot_tensor(inp, num_classes)
  local one_hot_val = torch.Tensor(inp:size(1), num_classes):zero()
  for i=1,inp:size(1) do
    one_hot_val[i][inp[i]] = 1
  end
  return one_hot_val
end

return utils

