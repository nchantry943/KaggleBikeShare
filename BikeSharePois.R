library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

## Upload Data
train <- vroom('train.csv')
test <- vroom('test.csv')

## Create Poisson Model
pois_mod <- poisson_reg() |>
  set_engine('glm') |>
  set_mode("regression") |>
  fit(count ~ factor(season) + factor(holiday) + factor(workingday) + factor(weather) + temp + humidity + windspeed, data = train)

bike_pois_pred <- predict(pois_mod, new_data = test)

pois_sub <- bike_pois_pred |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=pois_sub, file = "./PoisPreds.csv", delim=",")

