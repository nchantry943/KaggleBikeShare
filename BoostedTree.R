library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(poissonreg)
library(rpart)
library(ranger)
library(xgboost)

## Read in Data
train <- vroom('train.csv')
test <- vroom('test.csv')


## Clean Data 
train <- train |>
  select(-casual, -registered) |>
  mutate(count = log(count))

## Receipe
bike_rec <- recipe(count ~ ., data = train) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_time(datetime, features = c('hour')) |>
  step_rm(datetime) |>
  step_corr(all_predictors(), threshold = 0.75) |>
  step_mutate(season = factor(season)) |>
  step_mutate(holiday = factor(holiday)) |>
  step_mutate(workingday = factor(workingday)) |>
  step_mutate(weather = factor(weather)) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

prep_rec <- prep(bike_rec)
new_train <- bake(prep_rec, new_data = train)

##
model <- boost_tree(
  mode = "regression",
  engine = "xgboost",
  trees = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  stop_iter = tune()
)


## Workflow
workflow <- workflow() |>
  add_recipe(bike_rec) |>
  add_model(model)


grid <- grid_regular(trees(),
                     min_n(),
                     tree_depth(),
                     learn_rate(),
                     loss_reduction(),
                     stop_iter(),
                     levels = 3)

fold <- vfold_cv(train, v = 5, repeats = 1)

CV <- workflow |>
  tune_grid(resamples = fold, grid = grid, metrics = metric_set(rmse, mae, rsq))

best <- CV |> select_best(metric = 'rmse')

final_wf <- workflow |>
  finalize_workflow(best) |>
  fit(data = train)

lin_pred <- exp(predict(final_wf, new_data = test))

workflow_pred <- lin_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=workflow_pred, file = "./BoostedTreesPred.csv", delim=",")
