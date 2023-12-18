# Final Presentation
library(randomForest)
library(caret)
library(performance)
library(Metrics)
library(ggplot2)
library(factoextra)
library(SHAPforxgboost)
library(xgboost)
library(knitr)
library(shapviz)


country.data <- read.csv("~/6. DSMA/7. Seminar Data Science and Marketing Analytics/Final Document/world-data-2023.csv")
country.iq <- read.csv("~/6. DSMA/7. Seminar Data Science and Marketing Analytics/Final Document/avgIQpercountry.csv")

full.df <- merge(country.data, country.iq, by = "Country")

comma = function(column) {
  column = gsub(",", "", column)
  column = as.numeric(column)
  return(column)
}

percentt = function(column) {
  column = gsub("%", "", column)
  column = as.numeric(column)
  return(column)
}

usd = function(column) {
  column = gsub("$", "", column)
  column = as.numeric(column)
  return(column)
}

replace_blank_with_na <- function(dataset) {
  dataset <- as.data.frame(dataset)
  dataset[dataset == ""] <- NA
  return(dataset)
}


# ------------------------------- #
# -------- Data Cleaning -------- #
# ------------------------------- #
full.df.1 = full.df[,-c(1,3,8,9,13,21,25,34,35,36,38,44)]

full.df.1$Density..P.Km2. = gsub(",", "", full.df.1$Density..P.Km2.)
full.df.1$Density..P.Km2. = as.numeric(full.df.1$Density..P.Km2.)

full.df.1$Agricultural.Land.... = gsub("%", "", full.df.1$Agricultural.Land....)
full.df.1$Agricultural.Land.... = as.numeric(full.df.1$Agricultural.Land....)

full.df.1$Land.Area.Km2. = gsub(",", "", full.df.1$Land.Area.Km2.)
full.df.1$Land.Area.Km2. = as.numeric(full.df.1$Land.Area.Km2.)

full.df.1$Armed.Forces.size = gsub(",", "", full.df.1$Armed.Forces.size)
full.df.1$Armed.Forces.size = as.numeric(full.df.1$Armed.Forces.size)

full.df.1$Co2.Emissions = gsub(",", "", full.df.1$Co2.Emissions)
full.df.1$Co2.Emissions = as.numeric(full.df.1$Co2.Emissions)

full.df.1$CPI = as.numeric(full.df.1$CPI)

full.df.1$CPI.Change.... = gsub("%", "", full.df.1$CPI.Change....)
full.df.1$CPI.Change.... = as.numeric(full.df.1$CPI.Change....)

full.df.1$Forested.Area.... = percentt(full.df.1$Forested.Area....)

full.df.1$Gasoline.Price = gsub("\\$", "", full.df.1$Gasoline.Price)
full.df.1$Gasoline.Price = as.numeric(full.df.1$Gasoline.Price)

full.df.1$GDP = gsub("\\$", "", full.df.1$GDP)
full.df.1$GDP = gsub(",", "", full.df.1$GDP)
full.df.1$GDP = as.numeric(full.df.1$GDP)

full.df.1$Gross.primary.education.enrollment.... = percentt(full.df.1$Gross.primary.education.enrollment....)

full.df.1$Gross.tertiary.education.enrollment.... = percentt(full.df.1$Gross.tertiary.education.enrollment....)

full.df.1$Minimum.wage = gsub("\\$", "", full.df.1$Minimum.wage)
full.df.1$Minimum.wage = as.numeric(full.df.1$Minimum.wage)

full.df.1$Out.of.pocket.health.expenditure = percentt(full.df.1$Out.of.pocket.health.expenditure)

full.df.1$Population = gsub(",", "", full.df.1$Population)
full.df.1$Population = as.numeric(full.df.1$Population)

full.df.1$Population..Labor.force.participation.... = percentt(full.df.1$Population..Labor.force.participation....)

full.df.1$Tax.revenue....  = percentt(full.df.1$Tax.revenue.... )

full.df.1$Total.tax.rate  = percentt(full.df.1$Total.tax.rate )

full.df.1$Unemployment.rate  = percentt(full.df.1$Unemployment.rate )

full.df.1$Urban_population = gsub(",", "", full.df.1$Urban_population)
full.df.1$Urban_population = as.numeric(full.df.1$Urban_population)

x <- replace_blank_with_na(full.df.1)
x = na.omit(x)

# ------------------------------- #
# --------- Exploration --------- #
# ------------------------------- #
hist(x$Birth.Rate)
hist(x$Average.IQ)
hist(x$Density..P.Km2. )

for (i in 1:ncol(x)) {
  hist(x[,i], main =colnames(x)[i], xlab = "Distribution")
}
colnames(x[,2])
# ------------------------------- #
# ----- Data Transformation ----- #
# ------------------------------- #
x$GDP = log10(x$GDP)
x$CPI = log10(x$CPI)



# ------------------------------- #
# -------- Analysis -------- #
# ------------------------------- #
lm = lm(Average.IQ ~., data = x)
summary(lm)

rf = randomForest(Average.IQ ~ ., data = x)
varImpPlot(rf)
varUsed(rf)


# --------- # --------- #
# -- Xgb.Boost Model -- #
# --------- # --------- #

# Split the data into training and testing sets
index <- createDataPartition(y$Average.IQ, p = 0.8, list = FALSE)
train_data <- y[index, ]
test_data <- y[-index, ]

# Control for cross-validation
control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Define a grid
xgbGrid <- expand.grid(nrounds = c(100, 200), 
                       max_depth = c(3, 6, 9),
                       eta = c(0.01, 0.1, 0.3),
                       gamma = c(0, 0.1),
                       colsample_bytree = c(0.5, 1),
                       min_child_weight = c(1, 3, 5),
                       subsample = c(1, 0.5))

# Train the model using caret
set.seed(123)
xgbModel <- caret::train(Average.IQ ~ ., 
                         data = train_data, 
                         method = "xgbTree",
                         trControl = control,
                         tuneGrid = xgbGrid,
                         metric = "RMSE")

# Print the best model
print(xgbModel)

# First make model.matrix with same columns as the one used in caret finalModel
train_data.mat <- model.matrix(Average.IQ ~ . -1, data = train_data)

# Make SHAP values object
SHAP <- shapviz(xgbModel$finalModel, train_data.mat, interactions = TRUE)

# Showing plots
sv_importance(SHAP, show_numbers = TRUE)
sv_importance(SHAP, show_numbers = TRUE, kind = "bee")

sv_force(SHAP, row_id = 1)
sv_waterfall(SHAP, row_id = 20)

### Interaction plots
sv_interaction(SHAP, max_display = 3)


# Generate predictions for the test data
xgbTestPredictions <- predict(xgbModel, newdata = test_data)
y_acc = test_data$Average.IQ
data_res = data.frame(xgbTestPredictions, y_acc)

rmse = rmse(y_acc, xgbTestPredictions)

# Scatter plot of actual vs predicted values
ggplot(data_res, aes(x = y_acc, y = xgbTestPredictions)) + 
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Values", y = "Predicted Values", title = "Actual vs. Predicted Values")



# PCA
pca_df = y[,-29]

# Classify IQ
y$IQ_Class <- ifelse(y$Average.IQ >= 100, "High Intelligence",
                            ifelse(y$Average.IQ >= 90, "Above Average",
                                   ifelse(y$Average.IQ >= 80, "Average", 
                                          ifelse(y$Average.IQ >= 70, "Below Average",
                                          "Low Intelligence"))))

pca_result = prcomp(pca_df, scale. = TRUE)
fviz_pca_biplot(pca_result, axes = c(1,2), cex = 6,labelsize = 3,geom = "point",pointsize = 3,
                habillage = y$IQ_Class, palette = brewer)

fviz_contrib(pca_result, axes = 1, choice = "var")
fviz_contrib(pca_result, axes = 2, choice = "var")

fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 40))


