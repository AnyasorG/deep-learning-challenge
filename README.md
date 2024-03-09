# Module 21 Challenge - Alphabet Soup Charity Project

<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/accuracy_over_epochs_plot.png" alt="Before Optimization" />
  <p><em>Model Performance Before Optimization</em></p>
</div>

<div align="center">
  <img src="https://github.com/AnyasorG/deep-learning-challenge/blob/main/images/optimised_accuracy_over_epochs_plot.png" alt="After Optimization" />
  <p><em>Model Performance After Optimization</em></p>
</div>


## Overview:

The nonprofit foundation **Alphabet Soup** aims to develop a tool for selecting applicants with the best chance of success for funding. Utilizing machine learning and neural networks, the goal is to create a binary classifier predicting the success of organizations funded by Alphabet Soup. 
The provided dataset includes metadata columns such as EIN, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, and IS_SUCCESSFUL.

### Instructions:

**1. Google Colab:** Use Google Colab instead of Jupyter Notebook.

**2. Repository Setup:**
   - Create a new repository named `deep-learning-challenge` specifically for this project.
   - Clone the repository to your local machine.
   - Inside the local repository, create a directory dedicated to the Deep Learning Challenge.
   - Push the changes, including the new directory, to GitHub.

**3. Files:**
   - Download the necessary files for the challenge at https://github.com/AnyasorG/deep-learning-challenge.git
   - charity_data.csv  
   - AlphabetSoupCharity.ipynb 
   - AlphabetSoupCharity_Optimization.ipynb 
   - WrittenReportDeepLearningModel.md

## Project Development Stages:

### Step 1: Preprocess the Data

- Upload the starter file to Google Colab and complete the preprocessing steps.
- Read `charity_data.csv` into a Pandas DataFrame.
- Identify target variable(s) and feature variable(s).
- Drop the EIN and NAME columns.
- Determine the number of unique values for each column.
- For columns with over 10 unique values, identify data points for each unique value.
- Bin "rare" categorical variables together under a new value, "Other."
- Encode categorical variables using `pd.get_dummies()`.
- Split the preprocessed data into features array (X) and target array (y).
- Employ `train_test_split` to create training and testing datasets.
- Scale training and testing features using `StandardScaler`.

### Step 2: Compile, Train, and Evaluate the Model

- Utilize TensorFlow to design a neural network model.
- Assign the number of input features and nodes for each layer.
- Create hidden layers and an output layer with suitable activation functions.
- Check the model structure.
- Compile and train the model.
- Implement a callback to save model weights every five epochs.
- Evaluate the model using test data to determine loss and accuracy.
- Save and export results to an HDF5 file named `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

- Create a new Google Colab file named `AlphabetSoupCharity_Optimization.ipynb`.
- Import dependencies and read `charity_data.csv` into a Pandas DataFrame.
- Preprocess the dataset adjusting for any modifications from model optimization.
- Design a neural network model, optimizing it for accuracy above 75%.
- Save and export results to `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the optimised Deep Learning Model

- A written report on the Optimised Deep Learning Model is provided at https://github.com/AnyasorG/deep-learning-challenge.git. 

## Set Up and Execution:

Follow the instructions in the provided files for setting up the repository and executing the code.

## Ethical Considerations:

- All code is publicly available, promoting transparency in the analysis.

## Data Source:

- The dataset was located at IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/Links to an external site, provided by edX Boot Camps LLC and is intended for educational purposes only.

## Code Source:
- Parts of the code were adapted from documentation and resources, including:
  - [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
  - [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/)
  - [seaborn documentation](https://seaborn.pydata.org/documentation.html)
  - [tensorflow documentation](https://www.tensorflow.org/api_docs)


## License:

- This project is open-source and is made available under the terms of the MIT License. Refer to the [MIT License](https://choosealicense.com/licenses/mit/) for full details.

## Summary:

The project successfully develops and evaluates machine learning models for predicting the success of Alphabet Soup-funded organizations. 
Results and visualizations provide insights into model performance. 
See data at https://github.com/AnyasorG/deep-learning-challenge.git.

## Author:
Godswill Anyasor
