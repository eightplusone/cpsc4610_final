function loadData(fileLocation)
	local file = io.open(fileLocation)
	local ret = {}

	if file then
		local lineCounter = 1
		for line in file:lines() do
			table.insert(ret, split(line))
		end
	end
	return ret
end

function split(line)
	local arr = {}

	if line then
		for i = 1, 68 do
			local str = line:sub(i,i)
			if str ~= "\n" then
				table.insert(arr, str)
			end
		end
	return arr
	end
end

----------------------------------------------------------------------
-- Get the biggest value in a tensor
--
function getMax(values)
  local max = {}
	max[1] = 1
  for i = 2, 6 do
    if values[i] > values[max[1]] then max[1] = i end
  end
  return max
end
