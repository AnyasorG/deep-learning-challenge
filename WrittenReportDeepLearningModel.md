# Alphabet Soup Charity Project

## Overview:
The **Alphabet Soup Charity** project aimed to build and optimize a deep neural network model predicting whether applicants would be successful if funded by Alphabet Soup, with a target predictive accuracy higher than 75%. The dataset contains diverse organizational information, including application type, affiliation, classification, use case, organisation type, and financial details.

## Results: 

## Data Preprocessing:

### Target Variable:
- IS_SUCCESSFUL
  
<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/target_variable_distribution.png" alt="Target Variable" />
  <p><em>Figure 1: IS_SUCCESSFUL as Target Variable</em></p>
</div>


### Feature Variables:
- APPLICATION_TYPE
- AFFILIATION
- CLASSIFICATION
- USE_CASE
- ORGANISATION
- STATUS
- INCOME_AMT
- SPECIAL_CONSIDERATIONS
- ASK_AMT

<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/feature_variables_distribution_plot.png" alt="Feature Variables" />
  <p><em>Figure 2: Feature Variables</em></p>
</div>


### Removed Variables:
- EIN - Heat map shows a weak correlation between EIN and IS_SUCCESSFUL, hence its removal during preprocessing.
- NAME - Name is a non-binary categorical variables and was removal during preprocessing.

<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/correlation_heatmap_plot.png" alt="Removed Variables" />
  <p><em>Figure 3: Correlation matrix showing correlation coefficients between INCOME_AMT, ASK_AMT, and IS_SUCCESSFUL</em></p>
</div>


### Data Preparation Steps:
- Dropped non-beneficial columns 'EIN' and 'NAME'.'
- Applied one-hot encoding to categorical variables.
- Converted 'INCOME_AMT' to numeric format, handling different ranges and 'N' values.
  
<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/correlation_with_target_plot.png" alt="Correlation of Feature Variables with Target Variable" />
  <p><em>Figure 4: Correlation of feature variables with target variable</em></p>
</div>

## Compiling, Training, and Evaluating the Model:

### Model Architecture:
- first hidden layer:  80 units, ReLU activation
- second hidden layer: 30 units, ReLU activation
- Output layer:  1 unit, Sigmoid activation
- Epoch=100
- Accuracy: 72.41%
  
<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/model_architecture.png" alt="Model Architecture" />
  <p><em>Figure 5: Model architecture</em></p>
</div>


### Explanation:

- Neurons: The selection of neurons in each layer is crucial for capturing the complexity of the dataset. The chosen numbers were determined through experimentation, aiming for a balance between model complexity and generalization.

- Activation Functions:
    - ReLU (Rectified Linear Unit): Employed in hidden layers to introduce non-linearity and enable the model to learn intricate patterns.
    - Sigmoid: Applied in the output layer for binary classification, squeezing the output between 0 and 1 to represent the probability of success.

- Epochs: The model underwent training for 100 epochs, indicating the number of passes over the dataset during the learning process.

- Accuracy: The achieved accuracy of 72.41% reflects the model's effectiveness in correctly classifying instances. Further optimization may involve adjusting the architecture and hyperparameters based on the dataset's unique characteristics.

<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/accuracy_over_epochs_plot.png" alt="Model Accuracy Before Optimization" />
  <p><em>Figure 6: Model accuracy before optimization</em></p>
</div>

- The Training Accuracy Line (Blue) depicts the model's accuracy on the training dataset over epochs, indicating its learning progress.
- Validation Accuracy Line (Orange) shows the model's accuracy on a separate validation dataset, gauging its ability to generalize to new, unseen data.
- A higher Training Accuracy than Validation Accuracy implies potential overfitting, where the model excels on training data but struggles to generalize.



### Optimization Steps:
- Additional preprocessing step:  Dropped non-beneficial columns= ['EIN', 'NAME', 'STATUS', 'ASK_AMT', 'SPECIAL_CONSIDERATIONS'] as evidenced on the correlation heatmap as having coefficient values close to zero and below indicating very weak correlation in a range close to no correlation with IS_SUCCESSFUL.
- Multiple attempts adjusting hidden layers and neurons.
- Implemented early stopping to prevent overfitting.
- Adjusted the number of epochs for a balance between training time and performance.
  
<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/correlation_heatmap_plot.png" alt="Removed Variables" />
  <p><em>Figure 7: Correlation matrix showing correlation coefficients between 'EIN', 'STATUS', 'ASK_AMT', and IS_SUCCESSFUL. <strong>Note that 'NAME' and 'SPECIAL_CONSIDERATIONS' 
      are non-binary categorical variables.</strong></em></p>
</div>



<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/optimised_correlation_with_target_plot.png" alt="Optimized Correlation" />
  <p><em>Figure 8: Optimized correlation of feature variables with target variable</em></p>
</div>


### Final Optimized Architecture

<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/optimised_model_architecture.png" alt="Optimized Model Architecture" />
  <p><em>Figure 9: Optimized model architecture</em></p>
</div>

- First hidden layer:   150 units, leaky_relu activation  1st optimization method.
- Second hidden layer:  150 units, tanh activation        2nd optimization method.
- Third hidden layer:    50 units, relu activation        3rd optimization method.
- Output layer:           1 unit,  sigmoid activation
- Epoch=100                                               4th optimization method.
- Accuracy: 72.69%



<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/optimised_accuracy_over_epochs_plot.png" alt="Optimized Accuracy Over Epochs" />
  <p><em>Figure 10: Optimized model accuracy over epochs</em></p>
</div>


## Summary:
- The deep learning model, despite efforts to optimize the model, did not achieve the target accuracy of over 75%, reaching approximately 72.69%.
- However, the training accuracy(Blue) and validation accuracy (Orange) curves are closer to each other in the optimized model compared with the model before optimisation, which generally indicates that the model is not overfitting.
- Further exploration and tuning are needed for improvement.

## Recommendations:
- To address overfitting, consider techniques like regularization, dropout, or adjustments to the model architecture.
- Monitoring the gap between Training and Validation Accuracy is crucial for ensuring robust model performance on diverse datasets.
- Minimizing the gap is essential for building a model that generalizes well to new, unseen data.
- Experiment with Random Forest Classifier due to its robustness, ability to handle complex relationships, and resistance to overfitting, adjusting hyperparameters for optimal performance.
- The ensemble nature of Random Forests can capture non-linear patterns, potentially enhancing performance.
- Explore additional preprocessing techniques and feature engineering.
- Consider advanced techniques like gradient boosting.

## Conclusion:
The **Alphabet Soup Charity** project involved comprehensive data preprocessing, model building, and optimization. Despite efforts, achieving the target accuracy remains challenging, emphasizing the need for further refinement.
