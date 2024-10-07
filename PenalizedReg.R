library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(patchwork)
library(GGally)
library(car)
library(vroom)
library(glmnet)

## Read in Data
train <- vroom('train.csv')
test <- vroom('test.csv')

## Clean Data (Step 1)
train <- train |>
  select(-casual, -registered) |>
  mutate(count = log(count))

# Recipe setup
bike_pen_reg <- recipe(count ~ ., data = train) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_time(datetime, features = c('hour', 'minute')) |>
  step_rm(datetime) |>
  step_corr(all_predictors(), threshold = 0.75) |>
  step_mutate(season = factor(season)) |>
  step_mutate(holiday = factor(holiday)) |>
  step_mutate(workingday = factor(workingday)) |>
  step_mutate(weather = factor(weather)) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# Workflow setup and model training
pen_mod <- linear_reg(penalty = .1, mixture = .9) |> 
  set_engine("glmnet") |>
  set_mode("regression")

workflow <- workflow() |>
  add_recipe(bike_pen_reg) |>
  add_model(pen_mod) 

# Fit the model to the training data
fitted_workflow <- fit(workflow, data = train)

# Make predictions on the test set
lin_pred <- exp(predict(fitted_workflow, new_data = test))


## Formatting for Kaggle
workflow_pred <- lin_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=workflow_pred, file = "./PenalizedPred.csv", delim=",")



