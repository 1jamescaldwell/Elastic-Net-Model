This repository contains R code for predicting housing prices using elastic net regression. Here's a brief overview of the contents: <br>

* Data Preparation: The provided R script loads the training and test datasets (realestate-train.csv and realestate-test.csv respectively) and prepares them for modeling. It converts the CentralAir column to numerical, and performs dummy encoding for the HouseStyle and BldgType columns.

* Modeling: The main focus of the script is on fitting elastic net regression models using cross-validation. It iterates through different values of the alpha parameter (ranging from 0 to 1 by increments of 0.1) and fits elastic net models using each alpha value. The script then predicts housing prices on the test set and calculates the Root Mean Squared Error (RMSE) against the training set for each model. Additionally, it records the lambda parameter and the number of non-zero coefficients for each model.

* Model Evaluation and Selection: The script provides insights into the relationship between alpha, lambda, number of coefficients, and RMSE. It identifies an alpha value of 0.7 as the optimal choice based on the number of model parameters, as alpha values of 0.7 and above result in 17 coefficients, providing a balance between model complexity and predictive performance. The corresponding lambda for alpha = 0.7 is reported as 1.17.

* Prediction: Finally, the script generates predictions for housing prices on the test data using the selected alpha value (0.7) and writes the results to a CSV file named caldwell_james.csv.
