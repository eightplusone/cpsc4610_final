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

