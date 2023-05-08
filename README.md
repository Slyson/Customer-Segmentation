# Customer Segmentation Project

This project focuses on customer segmentation using various clustering and classification algorithms. The goal is to analyze customer data and categorize them into different segments based on their attributes.

## Dependencies

- scikit-learn
- numpy
- pandas
- matplotlib

## Dataset

The project utilizes the "customer-segmentation" dataset, which is loaded from the 'Train.csv' file. The dataset contains information about customers, including their gender, marital status, age, education, work experience, and segmentation category.

## Exploratory Data Analysis

- Gender Distribution: Visualizes the distribution of customers based on gender.
- Customers that ever married: Shows the distribution of customers based on their marital status.
- Age Distribution: Presents the distribution of customer ages.
- Graduated Customers: Displays the distribution of customers based on their education status.
- Customer Work Experience: Shows the distribution of customer work experience.

## Data Preprocessing

- Missing Value Handling: Missing values in the dataset are filled using the mean value for numeric columns and forward fill for other columns.
- Data Encoding: The 'Segmentation' column, representing the target variable, is label-encoded using sklearn's LabelEncoder. Categorical columns are one-hot encoded using OneHotEncoder.
- Data Scaling: The data is scaled using MinMaxScaler to ensure all features have a similar scale.

## Clustering Algorithms

- KMeans Clustering: Applies KMeans clustering algorithm with 4 clusters to the preprocessed data. Evaluates the clustering results using adjusted Rand index.
- Agglomerative Clustering: Performs Agglomerative Clustering with 4 clusters on the preprocessed data. Evaluates the clustering results using adjusted Rand index.

## Classification Algorithms

- Random Forest Classifier: Trains a Random Forest Classifier with 70 estimators and a maximum depth of 4 on the preprocessed data. Evaluates the classification performance using classification report and adjusted Rand index.
- K-Nearest Neighbors Classifier: Trains a K-Nearest Neighbors Classifier with 4 neighbors on the preprocessed data. Evaluates the classification performance using classification report.
- Gaussian Naive Bayes Classifier: Trains a Gaussian Naive Bayes Classifier on the preprocessed data. Evaluates the classification performance using classification report.
- Decision Tree Classifier: Trains a Decision Tree Classifier with a maximum depth of 4 and maximum leaf nodes of 10 on the preprocessed data. Evaluates the classification performance using classification report.
- MLP Classifier: Performs hyperparameter tuning using GridSearchCV on an MLP Classifier with adaptive learning rate and different hidden layer sizes and regularization parameters. Evaluates the classification performance using classification report.

Note: The project splits the data into training and test sets using train_test_split, and the evaluation metrics are reported on the test set.

Feel free to explore the project code and results to gain insights into customer segmentation using different algorithms.

## Usage

To run the project, make sure you have the required dependencies installed. You can execute the code in a Python environment, such as Jupyter Notebook or any Python IDE.

## License

This project is licensed under the [MIT License](LICENSE).
---
