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

----------------------------------------------------------------------
-- Configuration
-- Adjust these numbers if necessary
--
local INPUT_SIZE = 40    -- need confirmation
local HIDDEN_LAYER_SIZE = 6    -- play with this number
local OUTPUT_SIZE = 6
local LEARNING_RATE = 17e-3
local LEARNING_RATE_DECAY = 0
local WEIGHT_DECAY = 1e-2
local MOMENTUM = 9e-1
local LAMBDA = 0  -- for regularization
local MAX_EPOCH = 4e2  -- adjust base on the training result

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
-- Normal cases. I pre-define all 5 layers here, but your model may not need
-- to add them all.
local module_01 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
local module_02 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
local module_03 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
local module_04 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
local module_05 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

----------------------------------------------------------------------
-- Add modules to the neural network. Please do not uncomment both cases.
--
-- Special case: nets with no hidden layer
-- model:add(module_00)
--
-- Normal cases: please make sure you have the right number of hidden layers.
-- The template below represents a neural net with one hidden layer.
model:add(module_01)
-- model:add(module_02)
-- model:add(module_03)
-- model:add(module_04)
model:add(module_05)

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

  -- Need to define inputs here
  local inputs = ...

  -- Reset the gradients (by default, they are always accumulated)
  dl_dw:zero()

  -- Evaluate the loss function and its derivative with respect to w
  -- Step 1: Compute the prediction
  -- Step 2: Compute the loss (error)
  -- Step 3: Compute the gradient of the loss
  -- Step 4: Adjust the weights of the net
  local prediction = model:forward(inputs)
  local loss_w = criterion:forward(prediction, targets)
  local df_dw = criterion:backward(prediction, targets)
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
