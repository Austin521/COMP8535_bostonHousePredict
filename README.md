### A NeurIPs Format Report Outline: Boston Housing Price Prediction

#### Abstract
This report focuses on predicting Boston housing prices by exploring and comparing various data processing techniques, specifically dimensionality reduction using PCA and Kernel PCA (with a polynomial kernel). The report starts with a detailed description of each feature (independent variables) and the target variable (dependent variable) (Figure 1). Next, we handle missing values using the median imputation method. This approach is chosen because: 
1. **Robustness to outliers**: The median is not affected by outliers and better represents the central tendency of the data, maintaining the overall structure and distribution.
2. **Consistency and repeatability**: The median is a fixed value, ensuring consistent results each time missing values are imputed, thereby enhancing the interpretability and repeatability of data processing (Figure 2).

We then generate correlation heatmaps and scatter plots for each variable. Based on these plots (Figures 3 and 4), we suspect multicollinearity among the variables. This is confirmed by VIF tests, which show strong multicollinearity between some variables (with VIF values of 7 and 9) (Figure 5). To address this, we apply dimensionality reduction techniques to eliminate multicollinearity, aiming for a more stable model and more accurate predictions. 

We first use PCA, which reduces the dataset to two principal components that explain a significant portion of the variance (Number of features retained by PCA: 2, Variance ratio explained by PCA: [0.80582318, 0.16305197]). Then, we apply Kernel PCA (with a polynomial kernel), which also reduces the dataset to two principal components, explaining over 95% of the variance. The resulting plots (Figures 6 and 7) demonstrate that PCA captures the linear structure of the data, while Kernel PCA captures the nonlinear structure. The choice between these methods should be based on model performance. 

To obtain more accurate conclusions, we perform regression analysis on the data post-reduction and compare the results.

Finally, we evaluate the performance of these three data processing approaches using multiple linear regression and random forest regression (Figure 8):
1. **Performance on original data**:
   - **Linear Regression**: Performs best on the original data, with a test MSE of 24.291119 and a CV R-Squared of 0.698664.
   - **Random Forest**: Also performs very well on the original data, with a test MSE of 8.182469 and a CV R-Squared of 0.821278, indicating a strong linear structure in the data that both linear models and random forest can fit well.

2. **Performance after PCA reduction**:
   - **Linear Regression**: Shows poorer performance with a test MSE of 55.004638 and a CV R-Squared of 0.211555.
   - **Random Forest**: Similarly shows poorer performance with a test MSE of 49.505879 and a CV R-Squared of 0.283450.

3. **Performance after Kernel PCA reduction**:
   - **Linear Regression**: Performance is similar to PCA, with a test MSE of 58.762970 and a CV R-Squared of 0.187452.
   - **Random Forest**: Performance is again poorer compared to the original data, with a test MSE of 60.774121 and a CV R-Squared of 0.310851.

4. **Overfitting and underfitting considerations**:
   - **Underfitting**: Linear regression on PCA and Kernel PCA reduced data shows signs of underfitting, indicating these methods fail to capture the main features of the data. Random forest also shows underfitting, further indicating that reduced data does not effectively fit the model.
   - **Overfitting**: Random forest on the original data shows some overfitting, with very low training error but relatively higher test error, suggesting the model performs well on the training set but has reduced generalization ability on the test set.
   - **Ideal State**: Linear regression on the original data performs ideally, with close training and test errors, showing no significant overfitting or underfitting.

**Summary**:
1. **Data Structure Analysis**: Results indicate that regression models perform best on the original data, especially random forest. This suggests the data primarily exhibits a linear structure that linear models can fit well.
2. **Dimensionality Reduction Method Choice**: Although PCA and Kernel PCA have their advantages, they do not show significant improvements in model performance for this dataset. This suggests that dimensionality reduction does not provide a notable advantage in this context. Both PCA and Kernel PCA fail to retain essential information while reducing dimensions.
3. **Model Choice**: For predicting Boston housing prices, using the original data for modeling is recommended, particularly using random forest regression, which performs best on the original data.

**Further Improvements**:
1. **Explore more nonlinear features**: Although current analysis shows a linear structure, future work can explore more nonlinear features or advanced feature engineering techniques to further enhance model performance.
2. **Improve dimensionality reduction methods**: Try other dimensionality reduction methods or adjust the parameters of current methods to find a technique better suited for this dataset, thus improving model performance.

### Translation to English

### A NeurIPs Format Report Outline: Boston Housing Price Prediction

#### Abstract
This report focuses on predicting Boston housing prices by exploring and comparing various data processing techniques, specifically dimensionality reduction using PCA and Kernel PCA (with a polynomial kernel). The report starts with a detailed description of each feature (independent variables) and the target variable (dependent variable) (Figure 1). Next, we handle missing values using the median imputation method. This approach is chosen because: 
1. **Robustness to outliers**: The median is not affected by outliers and better represents the central tendency of the data, maintaining the overall structure and distribution.
2. **Consistency and repeatability**: The median is a fixed value, ensuring consistent results each time missing values are imputed, thereby enhancing the interpretability and repeatability of data processing (Figure 2).

We then generate correlation heatmaps and scatter plots for each variable. Based on these plots (Figures 3 and 4), we suspect multicollinearity among the variables. This is confirmed by VIF tests, which show strong multicollinearity between some variables (with VIF values of 7 and 9) (Figure 5). To address this, we apply dimensionality reduction techniques to eliminate multicollinearity, aiming for a more stable model and more accurate predictions. 

We first use PCA, which reduces the dataset to two principal components that explain a significant portion of the variance (Number of features retained by PCA: 2, Variance ratio explained by PCA: [0.80582318, 0.16305197]). Then, we apply Kernel PCA (with a polynomial kernel), which also reduces the dataset to two principal components, explaining over 95% of the variance. The resulting plots (Figures 6 and 7) demonstrate that PCA captures the linear structure of the data, while Kernel PCA captures the nonlinear structure. The choice between these methods should be based on model performance. 

To obtain more accurate conclusions, we perform regression analysis on the data post-reduction and compare the results.

Finally, we evaluate the performance of these three data processing approaches using multiple linear regression and random forest regression (Figure 8):
1. **Performance on original data**:
   - **Linear Regression**: Performs best on the original data, with a test MSE of 24.291119 and a CV R-Squared of 0.698664.
   - **Random Forest**: Also performs very well on the original data, with a test MSE of 8.182469 and a CV R-Squared of 0.821278, indicating a strong linear structure in the data that both linear models and random forest can fit well.

2. **Performance after PCA reduction**:
   - **Linear Regression**: Shows poorer performance with a test MSE of 55.004638 and a CV R-Squared of 0.211555.
   - **Random Forest**: Similarly shows poorer performance with a test MSE of 49.505879 and a CV R-Squared of 0.283450.

3. **Performance after Kernel PCA reduction**:
   - **Linear Regression**: Performance is similar to PCA, with a test MSE of 58.762970 and a CV R-Squared of 0.187452.
   - **Random Forest**: Performance is again poorer compared to the original data, with a test MSE of 60.774121 and a CV R-Squared of 0.310851.

4. **Overfitting and underfitting considerations**:
   - **Underfitting**: Linear regression on PCA and Kernel PCA reduced data shows signs of underfitting, indicating these methods fail to capture the main features of the data. Random forest also shows underfitting, further indicating that reduced data does not effectively fit the model.
   - **Overfitting**: Random forest on the original data shows some overfitting, with very low training error but relatively higher test error, suggesting the model performs well on the training set but has reduced generalization ability on the test set.
   - **Ideal State**: Linear regression on the original data performs ideally, with close training and test errors, showing no significant overfitting or underfitting.

**Summary**:
1. **Data Structure Analysis**: Results indicate that regression models perform best on the original data, especially random forest. This suggests the data primarily exhibits a linear structure that linear models can fit well.
2. **Dimensionality Reduction Method Choice**: Although PCA and Kernel PCA have their advantages, they do not show significant improvements in

 model performance for this dataset. This suggests that dimensionality reduction does not provide a notable advantage in this context. Both PCA and Kernel PCA fail to retain essential information while reducing dimensions.
3. **Model Choice**: For predicting Boston housing prices, using the original data for modeling is recommended, particularly using random forest regression, which performs best on the original data.

**Further Improvements**:
1. **Explore more nonlinear features**: Although current analysis shows a linear structure, future work can explore more nonlinear features or advanced feature engineering techniques to further enhance model performance.
2. **Improve dimensionality reduction methods**: Try other dimensionality reduction methods or adjust the parameters of current methods to find a technique better suited for this dataset, thus improving model performance.
