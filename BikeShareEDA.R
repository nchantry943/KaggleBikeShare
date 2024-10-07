library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(patchwork)
library(GGally)
library(car)
library(vroom)

## Upload Data
train <- vroom('train.csv')

## EDA
corrplot <- DataExplorer::plot_correlation(train)
glimpse(train)

pair <- ggpairs(train)
DataExplorer::plot_missing(train)
DataExplorer::plot_bar(train)
DataExplorer::plot_histogram(train)

weather <- ggplot(data = train, mapping = aes(x = weather)) + 
  geom_bar()

vif.lm <- lm(count ~ datetime + season + holiday + workingday + weather + temp + atemp + humidity + windspeed, data = train)
vifs <- vif(vif.lm)
vif_df <- data.frame(
  Variable = names(vifs),  
  VIF = as.numeric(vifs)   
)


gg_vif <- ggplot(data = vif_df, mapping = aes(x = reorder(Variable, VIF), y = VIF)) +
  geom_bar(stat = "identity") + 
  xlab('Variable')
gg_vif

temp <- ggplot(data = train, mapping = aes(x = temp, y = count)) + 
  geom_point() + 
  geom_smooth(se = FALSE)
temp
(corrplot + weather) / (gg_vif + temp)



