library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(patchwork)
library(GGally)
library(car)
library(vroom)
library(glmnet)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)

## Read in Data
train <- vroom('train.csv')
test <- vroom('test.csv')


## Clean Data 
train <- train |>
  select(-casual, -registered) |>
  mutate(count = log(count))

folds <- vfold_cv(train, v = 5, repeats = 1)


untuned <- control_stack_grid()
tuned <- control_stack_resamples()

## Penalized Regression
# preg_mod <- linear_reg(penalty = tune(),
#                        mixture = tune()) |>
#   set_engine('glmnet')
# 
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
# 
# prep_rec <- prep(bike_rec)
# new_train <- bake(prep_rec, new_data = train)
# 
# stack_wf <- workflow() |>
#   add_recipe(bike_rec) |>
#   add_model(preg_mod)
# 
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5)
# 
# preg_mod <- stack_wf |>
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(rmse, mae, rsq),
#             control = untuned)

## New Penalized
recipe_1 <- recipe(count~., data = train) %>%
  step_date(datetime, features = "dow") %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>%
  step_mutate(working_hour = workingday * datetime_hour) %>%
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
              holiday=factor(holiday),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy")))%>%
  step_mutate(datetime_hour=factor(datetime_hour)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())


preg_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")
preg_workflow <- workflow() %>%
  add_recipe(recipe_1) %>%
  add_model(preg_model)
preg_tuning_grid <- grid_regular(penalty(), mixture(), levels = 10)
preg_models <- preg_workflow %>%
  tune_grid(resamples=folds,
            grid=preg_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untuned)



## Random Forest
CV <- stack_wf |>
  tune_grid(resamples = folds, grid = tuning_grid, metrics = metric_set(rmse, mae, rsq))

best <- CV |> select_best(metric = 'rmse')

rf_wf <- stack_wf |>
  finalize_workflow(best) |>
  fit(data = train)

rf_mod <- fit_resamples(
  rf_wf,
  resamples = folds, metrics = metric_set(rmse, mae, rsq), control = tuned)

## Linear Regression
lin_mod <- linear_reg() |> 
  set_engine("lm") |>
  set_mode("regression") 

lin_reg_wf <- workflow() |>
  add_model(lin_mod) |>
  add_recipe(bike_rec)

lin_reg_mod <- 
  fit_resamples(
    lin_reg_wf,
    resamples = folds, 
    metrics = metric_set(rmse, mae, rsq), 
    control = tuned
  )


## Stack the predictions
my_stack <- stacks() |>
  add_candidates(preg_models) |>
  add_candidates(rf_mod) |>
  add_candidates(lin_reg_mod)

stack_mod  <- my_stack |>
  blend_predictions() |>
  fit_members()
  
stackdat <- as_tibble(my_stack)
stack_pred <- exp(predict(stack_mod, new_data=test))

## Finalize
workflow_pred <- stack_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=workflow_pred, file = "./StackingPred.csv", delim=",")


