library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(poissonreg)


## Read in Data
train <- vroom('train.csv')
test <- vroom('test.csv')


## Clean Data (Step 1)
train <- train |>
  select(-casual, -registered) |>
  mutate(count = log(count))


## Workflow
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


## Penalized Regression Model
p_reg_mod <- linear_reg(penalty = tune(),
                        mixture = tune()) |>
  set_engine('glmnet')

## Workflow
workflow <- workflow() |>
  add_recipe(bike_rec) |>
  add_model(p_reg_mod)

## Grid of values to tune
grid <- grid_regular(penalty(),
                     mixture(),
                     levels = 5)

## Cross Validation split
fold <- vfold_cv(train, v = 5, repeats = 1)

## Run CV
CV <- workflow |>
  tune_grid(resamples = fold, grid = grid, metrics = metric_set(rmse, mae, rsq))

## Plot Results
collect_metrics(CV) |>
  filter(.metric=='rmse') |>
  ggplot(aes(x=penalty, y= mean, color = factor(mixture))) +
  geom_line()

## Find Best tuning parameters
best <- CV |> select_best(metric = 'rmse')


## Finalize Workflow
final_wf <- workflow |>
  finalize_workflow(best) |>
  fit(data = train)

## Predict
lin_pred <- exp(predict(final_wf, new_data = test))


## Format
workflow_pred <- lin_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=workflow_pred, file = "./TuningPred.csv", delim=",")

