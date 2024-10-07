library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(poissonreg)
library(rpart)
library(ranger)
library(dbarts)

train <- vroom('train.csv')
test <- vroom('test.csv')


## Clean Data (Step 1)
train <- train |>
  select(-casual, -registered) |>
  mutate(count = log(count))

## Model
bart_mod <- parsnip::bart(trees = 100) |>
  set_engine('dbarts') |>
  set_mode('regression') |>
  translate()

## Receipe
bike_rec <- recipe(count ~ ., data = train) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_time(datetime, features = c('hour')) |>
  step_date(datetime, features = c('year', 'dow')) |>
  step_rm(datetime) |>
  step_mutate(int = workingday*datetime_hour) |>
  step_mutate(season = factor(season)) |>
  step_mutate(holiday = factor(holiday)) |>
  step_mutate(workingday = factor(workingday)) |>
  step_mutate(weather = factor(weather)) |>
  step_mutate(year = factor(datetime_year)) |>
  step_mutate(int = factor(int))

prep_rec <- prep(bike_rec)
new_train <- bake(prep_rec, new_data = train)

workflow <- workflow() |>
  add_recipe(bike_rec) |>
  add_model(bart_mod) |>
  fit(data = train)



lin_pred <- exp(predict(workflow, new_data = test))

workflow_pred <- lin_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=workflow_pred, file = "./BARTPred.csv", delim=",")


