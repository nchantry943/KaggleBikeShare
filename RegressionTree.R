library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(poissonreg)
library(rpart)
library(ranger)

## Read in Data
train <- vroom('train.csv')
test <- vroom('test.csv')


## Clean Data (Step 1)
train <- train |>
  select(-casual, -registered) |>
  mutate(count = log(count))

## Set up Trees
model <- decision_tree(tree_depth = tune(), 
                       cost_complexity = tune(),
                       min_n = tune()) |>
  set_engine('rpart') |>
  set_mode('regression')

## Create workflow with model and recipe
bike_rec <- recipe(count ~ ., data = train) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_time(datetime, features = c('hour')) |>
  step_rm(datetime) |>
  step_interact(~ workingday:datetime_hour)
  step_corr(all_predictors(), threshold = 0.75) |>
  step_mutate(season = factor(season)) |>
  step_mutate(holiday = factor(holiday)) |>
  step_mutate(workingday = factor(workingday)) |>
  step_mutate(weather = factor(weather)) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

prep_rec <- prep(bike_rec)
new_train <- bake(prep_rec, new_data = train)


## Workflow
workflow <- workflow() |>
  add_recipe(bike_rec) |>
  add_model(model)

## Grid of Tuning values and K-fold
grid <- grid_regular(tree_depth(),
                     cost_complexity(),
                     min_n(),
                     levels = 5)

fold <- vfold_cv(train, v = 5, repeats = 1)

CV <- workflow |>
  tune_grid(resamples = fold, grid = grid, metrics = metric_set(rmse, mae, rsq))

best <- CV |> select_best(metric = 'rmse')


## Finalize
final_wf <- workflow |>
  finalize_workflow(best) |>
  fit(data = train)

lin_pred <- exp(predict(final_wf, new_data = test))

workflow_pred <- lin_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=workflow_pred, file = "./RegressionTreesPred2.csv", delim=",")


## Random Forests
rand_for_mod <- rand_forest(mtry = tune(),
                            min_n = tune(), 
                            trees = 500) |>
  set_engine('ranger') |>
  set_mode("regression")

rand_work <- workflow() |>
  add_recipe(bike_rec) |>
  add_model(rand_for_mod)

grid1 <- grid_regular(mtry(range = c(1, 10)),
                     min_n(),
                     levels = 5)

fold1 <- vfold_cv(train, v = 5, repeats = 1)

CV1 <- rand_work |>
  tune_grid(resamples = fold1, 
            grid = grid1, 
            metrics = metric_set(rmse, mae, rsq))

best1 <- CV1 |> select_best(metric = 'rmse')

final_wf1 <- rand_work |>
  finalize_workflow(best1) |>
  fit(data = train)

for_pred <- exp(predict(final_wf1, new_data = test))

rand_pred <- for_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=rand_pred, file = "./RandomForestPred2.csv", delim=",")
