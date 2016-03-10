--[[
-- Author: Josh, Hai, Tyler
-- File: training_template.lua
--
-- This is the template for the neural network and the training process.
--]]

----------------------------------------------------------------------
-- Import
--
require 'nn'
require 'optim'
require 'gnuplot'
require('loadData.lua')

----------------------------------------------------------------------
-- Configuration
-- Adjust these numbers if necessary
--
local DATASET_SIZE = 90
local INPUT_SIZE = 67
local HIDDEN_LAYER_SIZE = 6    -- number of nodes per hidden layer. Play with this number: change it to 4, 6, 8, 10, 12
local OUTPUT_SIZE = 6
local LEARNING_RATE = 17e-3
local LEARNING_RATE_DECAY = 0
local WEIGHT_DECAY = 1e-2
local MOMENTUM = 9e-1
local MAX_EPOCH = 4e2  -- adjust base on the training result. Increase if the graph doesn't converge.
--local K_FOLD = 6  -- not implemented
--local BATCH_SIZE = 15  -- not implemented
local RUNS_PER_MODEL = 5  -- 5 runs for each model
local DATAFILE = '../data.txt'

----------------------------------------------------------------------
-- Decalre the neural network. By default, we only play with
-- sequential nets.
--
local model = nn.Sequential()

----------------------------------------------------------------------
-- List of modules
--
-- Special case: nets with no hidden layer
local module_00 = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
--
-- Normal cases. I pre-define all 7 layers here, but your model may not need
-- all of them.
local module_01 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
local module_02 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
local module_03 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
local module_04 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
local module_05 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
local module_06 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
local module_07 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
--
-- NOTE: We have this layer below because Torch keeps complaining about
-- the size of the tensors during backpropagation. It does not count toward
-- the total number of hidden layers
local module_out = nn.Linear(OUTPUT_SIZE, 1)

----------------------------------------------------------------------
-- Add modules to the neural network. Please do not uncomment both cases.
--
-- Special case: nets with no hidden layer
-- NOTE: This model needs another implementation for the training step
-- due the the mismatch tensor size. We will skip it this time.
--model:add(module_00)
--
-- PLEASE ADJUST THIS NUMBER FIRST
local NUM_HIDDEN_LAYERS = 6
--
-- Normal cases: please make sure you have the right number of hidden layers.
-- The template below represents a neural net with six hidden layer.
model:add(module_01)
model:add(module_02)
model:add(module_03)
model:add(module_04)
model:add(module_05)
model:add(module_06)
model:add(module_07)
model:add(module_out)

----------------------------------------------------------------------
-- Explanation from https://raw.githubusercontent.com/andresy/torch-demos/master/linear-regression/example-linear-regression.lua
-- Define a loss function, to be minimized.
-- In that example, we minimize the Mean Square Error (MSE) between
-- the predictions of our linear model and the groundtruth available
-- in the dataset.
-- Torch provides many common criterions to train neural networks.
--
local criterion = nn.MSECriterion()
criterion.sizeAverage = false

----------------------------------------------------------------------
-- Prepare data
-- We use 6-fold cross-validation technique here.
-- With 90 entries in the dataset, we shuffle and divide them into 6 subsets,
-- each subset contains 15 entries.
-- In each subset, we pick 9 entries for the training set, 3 for
-- validation, and 3 for testing.
--
-- UPDATE: We will not implement k-fold. We will just simple use 1-fold:
-- 60% for training set, 20% for validation set, and 20% for test set.
local data = loadData(DATAFILE)
--
-- Generate a random permutation of a sequence between 1 and DATASET_SIZE
local indices = torch.randperm(DATASET_SIZE)
--
-- Split data
local trainSet, validationSet, testSet = {}, {}, {}
for i = 1, DATASET_SIZE*0.6 + 1 do
  table.insert(trainSet, data[indices[i]])
end
for i = DATASET_SIZE*0.6 + 1, DATASET_SIZE*0.8+1 do
  table.insert(validationSet, data[indices[i]])
end
for i = DATASET_SIZE*0.8 + 1, DATASET_SIZE do
  table.insert(testSet, data[indices[i]])
end
trainSet = torch.Tensor(trainSet)
validationSet = torch.Tensor(validationSet)
testSet  = torch.Tensor(testSet)

----------------------------------------------------------------------
-- Training function
-- Reference: http://rnduja.github.io/2015/10/26/deep_learning_with_torch_step_7_optim/
--
-- Explanation from https://raw.githubusercontent.com/andresy/torch-demos/master/linear-regression/example-linear-regression.lua
-- To minimize the loss defined above, using the linear model defined
-- in 'model', we follow a stochastic gradient descent procedure (SGD).
-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the
-- entire training set is too costly.
-- Given an arbitrarily complex model, we can retrieve its trainable
-- parameters, and the gradients of our loss function wrt these
-- parameters by doing so:
w, dl_dw = model:getParameters()
--
-- The SGD (Stochastic Gradient Descent) optimization function in the
-- optim library has the signature of:
--     optim.sgd(func,x,state)
-- Normally, people change the name of "func" to "opfunc" to avoid duplication.
-- The below feval function computes the value of the loss function at a
-- given point w, and the gradient of that function with respect to w.
-- w is the vector of trainable weights, which are all the weights of
-- the linear matrix of our net, plus one bias.
-- w_new is the updated weights.
opfunc = function(w_new)
  -- Copy the weights if they were updated in the last iteration
  -- They are vectors (or Tensors in the world of Torch), so we need to use
  -- copy() funciton
  if w ~= w_new then
    w:copy(w_new)
  end

  -- select a new training sample
  _nidx_ = (_nidx_ or 0) + 1
  if _nidx_ > (#trainSet)[1] then _nidx_ = 1 end

  local sample = trainSet[_nidx_]
  local target = sample[{ {INPUT_SIZE+1} }]    -- this funny looking syntax allows
  local inputs = sample[{ {1,INPUT_SIZE} }]    -- slicing of arrays.

  -- Reset the gradients (by default, they are always accumulated)
  dl_dw:zero()

  -- Evaluate the loss function and its derivative with respect to w
  -- Step 1: Compute the prediction
  -- Step 2: Compute the loss (error)
  -- Step 3: Compute the gradient of the loss
  -- Step 4: Adjust the weights of the net
  local prediction = model:forward(inputs)
  local loss_w = criterion:forward(prediction, target)
  local df_dw = criterion:backward(prediction, target)
  model:backward(inputs, df_dw)

  -- return loss and its derivative
  return loss_w, dl_dw
end

----------------------------------------------------------------------
-- Validate function
--
fval = function()
  -- Set model to non-training mode
  model:evaluate()

  -- Store the cumulatedLoss of the validation set
  local cumulatedLoss = 0
  -- Loop through validation set
  for i = 1, DATASET_SIZE*0.2 + 1 do
    -- Load validation data
    local sample = trainSet[i]
    local target = sample[{ {INPUT_SIZE+1} }]    -- this funny looking syntax allows
    local inputs = sample[{ {1,INPUT_SIZE} }]    -- slicing of arrays.
    -- Calculate the loss
    local prediction = model:forward(inputs)
    local loss_w = criterion:forward(prediction, target)
    -- Add to the cumulatedLoss
    cumulatedLoss = cumulatedLoss + loss_w
  end

  -- Set model back to training mode
  model:training()

  return cumulatedLoss/(DATASET_SIZE*0.2)
end

----------------------------------------------------------------------
-- Test function
--
ftest = function()
  local correct = 0
  for i = 1, DATASET_SIZE*0.2 do
    -- Get inputs & target label
    local sample = testSet[i]
    local target = sample[{ {INPUT_SIZE+1} }]    -- this funny looking syntax allows
    local inputs = sample[{ {1,INPUT_SIZE} }]    -- slicing of arrays.
    -- Feed into network
    local prediction = model:forward(inputs)
    -- Check if prediction == target
    local predictedLabel = math.floor(prediction[1] + 0.5)
    if predictedLabel == target[1] then
      correct = correct + 1
    end
  end
  return (correct*100)/(DATASET_SIZE*0.2)
end

----------------------------------------------------------------------
-- Explanation from https://raw.githubusercontent.com/andresy/torch-demos/master/linear-regression/example-linear-regression.lua
-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely
--
state = {
  learningRate = LEARNING_RATE,
  learningRateDecay = LEARNING_RATE_DECAY,
  weightDecay = WEIGHT_DECAY,
  momentum = MOMENTUM
}

----------------------------------------------------------------------
-- Run
--
for run = 1, RUNS_PER_MODEL do
  -- Start a timer
  local timer = torch.Timer()
  -- Store the error of all epoches
  local trainingLoss = {}
  -- Store validation accuracy of all epoches
  local validationLoss = {}
  -- For early stopping
  local bestAccuracy, bestEpoch = 20, 0
  -- Save the best model into a file
  local modelFile = '../model/model_' .. NUM_HIDDEN_LAYERS .. '_' .. HIDDEN_LAYER_SIZE

  for epoch = 1, MAX_EPOCH do

    -- This variable is used to estimate the average loss
    trainingLoss[epoch] = 0

    -- An epoch is a full loop over our training data
    for j = 1, DATASET_SIZE*0.6 + 1 do

      -- The optim library contains several optimization algorithms.
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to w,
      --     given a point w
      --   + a point w
      --   + state of the net, which are algorithm-specific
      -- Functions in optim all return two things:
      --   + the new w, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.
      -- (w = weights)
      w_new, fs = optim.sgd(opfunc, w, state)
      -- Accumulate the loss
      trainingLoss[epoch] = trainingLoss[epoch] + fs[1]
    end

    -- Report average error on epoch
    trainingLoss[epoch] = trainingLoss[epoch] / DATASET_SIZE
    print('Training loss = ' .. trainingLoss[epoch])

    -- Validation
    validationLoss[epoch] = fval()
    print('Validation loss = ' .. validationLoss[epoch])

    -- Early Stopping when no new maxima has been found in 10 epoches
    local wait = 0
    if epoch % 50 == 0 then
      if trainingLoss[epoch] > trainingLoss[epoch-1] and
          validationLoss[epoch] < bestAccuracy then
        bestAccuracy, bestEpoch = validationLoss[epoch], epoch
        torch.save(modelFile, model)
        wait = 0
      else
        break
      end
    end

  end

  -- Get training time
  local trainingTime = timer:time().real
  print('Training time: ' .. trainingTime .. ' seconds')

  -- Test
  testAccuracy = ftest()
  print('Test accuracy: ' .. testAccuracy)

  -- Plot
  local graphFile = '../result/graph_' .. NUM_HIDDEN_LAYERS .. '_' .. HIDDEN_LAYER_SIZE .. '_' .. run .. '.png'
  gnuplot.pngfigure(graphFile)
  gnuplot.title('Training & Validation Loss')
  gnuplot.ylabel('Loss')
  gnuplot.xlabel('Epoch (x5)')
  -- Select data to plot (to make the graph clearer)
  local tLoss, vLoss = {}, {}
  for i = 1, #trainingLoss do
    if i%5==0 then
      tLoss[math.floor(i/5)] = trainingLoss[i]
      vLoss[math.floor(i/5)] = validationLoss[i]
    end
  end
  -- Add two lists to the drawing
  gnuplot.plot(
      {'Train Loss', torch.Tensor(tLoss)},
      {'Validation Loss', torch.Tensor(vLoss)}
  )
  gnuplot.plotflush()

  -- Log result
  --
  local logFile = '../result/result_' .. NUM_HIDDEN_LAYERS .. '_' .. HIDDEN_LAYER_SIZE .. '.txt'
  local file, fileErr = io.open(logFile, 'a+')
  if fileErr then print('File Open Error')
  else
    file:write('\n----- Run #' .. run .. ' -----\n')
    file:write('Duration: ' .. trainingTime .. '\n')
    file:write('Test Accuracy: ' .. testAccuracy .. '\n')
    file:close()
  end
end
