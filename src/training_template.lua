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
local HIDDEN_LAYER_SIZE = 10    -- play with this number
local OUTPUT_SIZE = 6
local LEARNING_RATE = 17e-3
local LEARNING_RATE_DECAY = 0
local WEIGHT_DECAY = 1e-2
local MOMENTUM = 9e-1
local MAX_EPOCH = 4e2  -- adjust base on the training result
local K_FOLD = 6
local BATCH_SIZE = 15
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
local data = loadData(DATAFILE)
data = torch.Tensor(data)
local input = data[{ {INPUT_SIZE+1} }]
local labels = data[{ {1,INPUT_SIZE} }]
--
-- Generate a random permutation of a sequence between 1 and DATASET_SIZE
local indices = torch.randperm(DATASET_SIZE)

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
-- Normally, people change the name of "func" to "feval" to avoid duplication.
-- The below feval function computes the value of the loss function at a
-- given point w, and the gradient of that function with respect to w.
-- w is the vector of trainable weights, which are all the weights of
-- the linear matrix of our net, plus one bias.
-- w_new is the updated weights.
feval = function(w_new)
  -- Copy the weights if they were updated in the last iteration
  -- They are vectors (or Tensors in the world of Torch), so we need to use
  -- copy() funciton
  if w ~= w_new then
    w:copy(w_new)
  end

  -- select a new training sample
  _nidx_ = (_nidx_ or 0) + 1
  if _nidx_ > (#data)[1] then _nidx_ = 1 end

  local sample = data[_nidx_]
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
-- Training
--
timer = torch.Timer()

for i = 1, MAX_EPOCH do
  -- This variable is used to estimate the average loss
  currentLoss = 0

  -- An epoch is a full loop over our training data
  for i = 1, DATASET_SIZE*0.6 do

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
    w_new, fs = optim.sgd(feval, w, state)

    currentLoss = currentLoss + fs[1]
  end

  -- report average error on epoch
  currentLoss = currentLoss / DATASET_SIZE
  print('current loss = ' .. currentLoss)
end

print('Time elapsed: ' .. timer:time().real .. ' seconds')

----------------------------------------------------------------------
-- Testing
prediction = model:forward(data[19][{ {1,INPUT_SIZE} }])
x = math.floor(prediction[1] + 0.5)
print(x)

----------------------------------------------------------------------
-- Log result
--
local filename = '../result' .. NUM_HIDDEN_LAYERS .. HIDDEN_LAYER_SIZE .. ''
