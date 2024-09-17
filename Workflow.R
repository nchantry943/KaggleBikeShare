library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(patchwork)
library(GGally)
library(car)
library(vroom)


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
  step_time(datetime, features = c('hour', 'minute')) |>
  step_rm(datetime) |>
  step_corr(all_predictors(), threshold = 0.75) |>
  step_mutate(season = factor(season)) |>
  step_mutate(holiday = factor(holiday)) |>
  step_mutate(workingday = factor(workingday)) |>
  step_mutate(weather = factor(weather)) |>
  step_mutate(datetime_hour = factor(datetime_hour))

prep_rec <- prep(bike_rec)
new_train <- bake(prep_rec, new_data = train)

lin_mod <- linear_reg() |> 
  set_engine("lm") |>
  set_mode("regression")

workflow <- workflow() |>
  add_recipe(bike_rec) |>
  add_model(lin_mod) |>
  fit(data = train)

lin_pred <- exp(predict(workflow, new_data = test))

## Formatting for Kaggle
workflow_pred <- lin_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=workflow_pred, file = "./WorkflowPreds.csv", delim=",")



