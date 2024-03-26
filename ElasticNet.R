# March 2024
# 
# This R script predicts housing prices using elastic net regression. Here's a brief overview of the contents:
# 
# Data Preparation: The provided R script loads the training and test datasets (realestate-train.csv and realestate-test.csv respectively) and prepares them for modeling. It converts the CentralAir column to numerical, and performs dummy encoding for the HouseStyle and BldgType columns.
# 
# Modeling: The main focus of the script is on fitting elastic net regression models using cross-validation. It iterates through different values of the alpha parameter (ranging from 0 to 1 by increments of 0.1) and fits elastic net models using each alpha value. The script then predicts housing prices on the test set and calculates the Root Mean Squared Error (RMSE) against the training set for each model. Additionally, it records the lambda parameter and the number of non-zero coefficients for each model.
# 
# Model Evaluation and Selection: The script provides insights into the relationship between alpha, lambda, number of coefficients, and RMSE. It identifies an alpha value of 0.7 as the optimal choice based on the number of model parameters, as alpha values of 0.7 and above result in 17 coefficients, providing a balance between model complexity and predictive performance. The corresponding lambda for alpha = 0.7 is reported as 1.17.
# 
# Prediction: Finally, the script generates predictions for housing prices on the test data using the selected alpha value (0.7) and writes the results to a CSV file named caldwell_james.csv.



library(mlbench)
library(glmnet)
library(tidymodels)# for optional tidymodels solutions
library(tidyverse) # functions for data manipulation 

# Load data
rs_train <- read.csv("realestate-train.csv", header = TRUE)
rs_test <- read.csv("realestate-test.csv", header = TRUE)

# Convert CentralAir column to numerical
rs_train$CentralAir <- ifelse(rs_train$CentralAir == "Y", 1, 0)
rs_test$CentralAir <- ifelse(rs_test$CentralAir == "Y", 1, 0)

# Convert HouseStyle to numerical (dummy encoding)
rs_train$HouseStyle <- as.factor(rs_train$HouseStyle)
encoded <- model.matrix(~HouseStyle - 1, data = rs_train)
rs_train <- cbind(rs_train, encoded)

# Convert BldgType to numerical (dummy encoding)
rs_train$BldgType <- as.factor(rs_train$BldgType)
encoded <- model.matrix(~BldgType - 1, data = rs_train)
rs_train <- cbind(rs_train, encoded)

# Convert HouseStyle to numerical (dummy encoding)
rs_test$HouseStyle <- as.factor(rs_test$HouseStyle)
encoded <- model.matrix(~HouseStyle - 1, data = rs_test)
rs_test <- cbind(rs_test, encoded)

# Convert BldgType to numerical (dummy encoding)
rs_test$BldgType <- as.factor(rs_test$BldgType)
encoded <- model.matrix(~BldgType - 1, data = rs_test)
rs_test <- cbind(rs_test, encoded)

# print(head(rs_train,10))

#Make x/y train and x/y test

#Exclude columns: price, BldgType, and HouseStyle
X.train <- rs_train[, !(names(rs_train) %in% c("price", "BldgType", "HouseStyle"))]
Y.train <- rs_train[,"price"]

X.test <- rs_test[, !(names(rs_test) %in% c("BldgType", "HouseStyle"))]

X.train <- as.matrix(X.train)
X.test <- as.matrix(X.test)
Y.train <- as.matrix(Y.train)

# My solution here conducts elastic net regression using cross-validation to predict housing prices. The data is split into 10 folds for cross-validation. I wasn't sure which alpha value to choose, so I have the script iterate through different values of the alpha parameter (0 to 1 by 0.1 increments) and fits elastic net models using each alpha value using cv.glmnet. For each model, it predicts housing prices on the test set and calculates RMSE against the training set. The script also records the lambda parameter and the number of non-zero coefficients for each model. Finally, it prints the RMSE, lambda, and number of non-zero coefficients for each alpha value.

# Set seed for reproducibility
set.seed(22)

# Initialize lists to store results
yhat.enet_list <- list()
yhat.enet_train_list <- list()
num_coeff_list <- list()
rmse_list <- list()
lambda_list <- list()

n.folds = 10 # number of folds for cross-validation
fold = sample(rep(1:n.folds, length=nrow(X.train)))

# Values of alpha (a)
alpha_values <- seq(0, 1, by = 0.1)

# Loop through alpha values
for (a in alpha_values) {
  # Set alpha for elastic net
  fit.enet <- cv.glmnet(X.train, Y.train, alpha = a, foldid = fold)
  beta.enet <- coef(fit.enet, s = "lambda.min")
  
  # Predictions for test data
  yhat.enet <- predict(fit.enet, newx = X.test, s = "lambda.min")
  yhat.enet_list[[as.character(a)]] <- yhat.enet
  
  # Predictions for train data
  yhat.enet_train <- predict(fit.enet, newx = X.train, s = "lambda.min")
  yhat.enet_train_list[[as.character(a)]] <- yhat.enet_train
  
  lambda_list[[as.character(a)]] <- fit.enet$lambda.min
  
  # Number of non-zero coefficients
  num_coeff <- length(which(beta.enet@x != 0))
  num_coeff_list[[as.character(a)]] <- num_coeff
  
  # Calculate RMSE
  residuals_train <- Y.train - yhat.enet_train
  RMSE_train <- sqrt(mean(residuals_train^2))
  rmse_list[[as.character(a)]] <- RMSE_train
}

print(rmse_list)
print(lambda_list)
print(num_coeff_list)



# As the alpha value increases from 0 to 0.1, there is a decrease in the RMSE. However, beyond an alpha value of 0.1, the RMSE remains relatively constant at approximately 38.8, while the lambda values decrease. Having an alpha = .1 will be a more complex model and possibly overfit to the training data. Since the RMSE values are similar for most alphas, I have decided to choose an alpha value based on the # of model parameters. alpha = .7 and above all have 17 coefficients. Thus, I've decided to use a = .7. The corresponding lambda for a = .7 is 1.17. 

yhat_alpha_07 <- yhat.enet_list[["0.7"]]

# Convert to data frame
yhat <- as.data.frame(yhat_alpha_07)
names(yhat) <- "yhat"
# Write to CSV
write.csv(yhat, file = "caldwell_james.csv", row.names = FALSE)


# The predicted price of the training data evaluated against the actual price of the training data has the following RMSE:

print(rmse_list[["0.7"]])
