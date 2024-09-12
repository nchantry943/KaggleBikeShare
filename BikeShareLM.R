library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(patchwork)
library(GGally)
library(car)
library(vroom)

## Upload Data
train <- vroom('train.csv')
test <- vroom('test.csv')

## Create Linear Model
train$weather[train$weather == 4] <- 3 ## Grouping the value where weather = 4 to be in the group weather = 3
test$weather[test$weather == 4] <- 3

lin_mod <- linear_reg() |> 
  set_engine("lm") |>
  set_mode("regression") |>
  fit(count ~ factor(season) + factor(holiday) + factor(workingday) + factor(weather) + temp + humidity + windspeed, data = train)


preds <- predict(lin_mod, new_data = test)

sub <- preds |>
  bind_cols(test) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=sub, file = ".LinearPreds.csv", delim=",")
