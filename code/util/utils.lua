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

return utils

