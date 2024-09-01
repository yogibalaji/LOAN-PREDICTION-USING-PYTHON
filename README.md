#LOAN ELIGIBILITY PREDICTION USING PYTHON 

Loan Prediction Analysis and Model Training
Overview
This project involves analyzing a customer dataset related to loan applications, performing data preprocessing, exploratory data analysis, and training a logistic regression model to predict the approval of a loan. The dataset includes various features like gender, marital status, education, income, loan amount, and more. The project is divided into several steps including data cleaning, visualization, feature encoding, model training, and evaluation.

Prerequisites
Before running the code, ensure that you have the following Python libraries installed:

numpy
pandas
matplotlib
seaborn
scikit-learn
You can install these dependencies using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn
Dataset
The dataset used in this project is stored in a CSV file named CUSTOMERS.csv. The dataset contains various features about customers applying for loans, including:

Gender: Gender of the applicant (Male, Female)
Married: Marital status (Yes, No)
Dependents: Number of dependents (0, 1, 2, 3+)
Education: Education level (Graduate, Not Graduate)
Self_Employed: Self-employment status (Yes, No)
ApplicantIncome: Income of the applicant
CoapplicantIncome: Income of the co-applicant
LoanAmount: Loan amount requested
Loan_Amount_Term: Term of the loan (in months)
Credit_History: Credit history (1, 0)
Property_Area: Area of property (Urban, Semiurban, Rural)
Loan_Status: Status of the loan application (Y/N)
Project Steps
1. Data Loading
The dataset is loaded into a Pandas DataFrame, and basic information about the data such as the first few rows, shape, and missing values is displayed.

2. Data Preprocessing
Missing Values: Missing values in the LoanAmount and Credit_History columns are filled with the mean and median, respectively. Remaining rows with missing values are dropped.
Categorical Encoding: Categorical variables are encoded into numerical values to be used in the model. For instance, Gender is encoded as 1 for Male and 0 for Female.
3. Data Visualization
Users can select from a range of visualizations to explore relationships between features and loan status:

Loan Status by Gender: Visualizes the distribution of loan status based on gender.
Loan Status by Marital Status: Visualizes the distribution of loan status based on marital status.
Loan Status by Education: Visualizes the distribution of loan status based on education.
Loan Status by Self Employment: Visualizes the distribution of loan status based on self-employment status.
Loan Status by Property Area: Visualizes the distribution of loan status based on the area of the property.
Distribution of Loan Amount: Plots the distribution of the loan amount.
Distribution of Applicant Income: Plots the distribution of the applicant's income.
Distribution of Coapplicant Income: Plots the distribution of the co-applicant's income.
Loan Amount by Property Area: Displays a box plot of loan amounts across different property areas.
Correlation Heatmap: Shows the correlation between numerical features.
Pairplot of Features: Visualizes relationships between different features using pair plots.
4. Model Training
Feature Standardization: Features are standardized using StandardScaler to improve model performance.
Train-Test Split: The dataset is split into training and testing sets (70%-30% split).
Logistic Regression Model: A logistic regression model is trained on the training set to predict loan approval.
5. Model Evaluation
Accuracy Score: The accuracy of the model is calculated using the test set.
Classification Report: Provides detailed metrics like precision, recall, and F1-score.
Confusion Matrix: Visualizes the performance of the model in terms of true positives, false positives, true negatives, and false negatives.
ROC Curve: Plots the Receiver Operating Characteristic curve to visualize the trade-off between true positive and false positive rates.
Feature Importance: Displays the importance of each feature in predicting the loan status, based on the coefficients from the logistic regression model.
How to Run the Code
Ensure the dataset (CUSTOMERS.csv) is placed in the same directory as the Python script.
Run the script using a Python interpreter.
Follow the prompts to select the graphs you wish to visualize.
The script will automatically train the logistic regression model and display the evaluation metrics.
License
This project is open-source and available under the MIT License.
