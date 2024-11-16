Sales Analysis and Prediction Using AI
This project demonstrates the use of machine learning to analyze and predict sales data using a simple linear regression model. The analysis and prediction are implemented with the help of Streamlit for the web interface and several popular Python libraries like Pandas, Matplotlib, Seaborn, and Scikit-learn.

Project Overview
This project allows users to interact with data, perform initial analysis, and predict sales prices based on certain features (e.g., age and passenger class in the Titanic dataset). Users can explore the data's statistical details, visualize key trends, and see how different variables influence the prediction of sales or fares.

Technologies Used
Streamlit: Used for creating the interactive web interface to display data, visualizations, and predictions.
Pandas: Utilized for data manipulation and analysis.
Matplotlib & Seaborn: For visualizing the data and generating plots like histograms and count plots.
Scikit-learn: Implements the machine learning model (linear regression) and tools for splitting data and evaluating model performance.
How It Works
Data Loading:

The dataset is loaded from an external URL. (In this example, it's the Titanic dataset from Kaggle, but it can be swapped with your own sales dataset in CSV format).
Initial Data Exploration:

The app displays the first few rows of the data and provides basic statistics (like mean, median, standard deviation) for numeric columns.
Data Visualization:

The application displays two visualizations:
A histogram showing the distribution of ticket fares.
A count plot displaying the number of passengers in each class.
Data Preprocessing for Model Training:

Missing values are handled by dropping rows with missing 'Age' or 'Fare' values.
A feature set is selected (Age and Pclass) to predict the target variable (Fare).
Model Training and Evaluation:

The dataset is split into training and test sets.
A simple linear regression model is trained on the training data.
The model's performance is evaluated using the Mean Squared Error (MSE), and the coefficients of the model are displayed.
Prediction with New Input:

Users can input their own values for age and passenger class using Streamlit widgets.
The model then predicts the fare (ticket price) based on the user's inputs.
Key Features
Data Visualization: Interactive charts to analyze the data and understand patterns.
Machine Learning: The app uses a simple linear regression model to predict sales or fare values.
User Input for Prediction: Allows users to interactively input data (like age and class) and get predictions.
Installation & Setup
To run this project locally, follow these steps:

Clone this repository:

bash
Copy code
git clone https://github.com/your-username/sales-prediction.git
cd sales-prediction
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Access the app in your browser at http://localhost:8501.

Customization
To use your own sales dataset, modify the URL in the load_data() function to point to your data file (CSV file or dataset URL).
You can also modify the features and target variable used in the model training (currently, it's using 'Age' and 'Pclass' to predict 'Fare').
Example of Expected Output
Statistical summary of the data.
Visualizations of the ticket fare distribution and passenger class distribution.
The model's coefficients and prediction accuracy (Mean Squared Error).
Interactive widgets for predicting ticket fares based on user input.
Future Improvements
Implement more sophisticated machine learning models for better predictions (e.g., Random Forest, XGBoost).
Add more detailed data cleaning and preprocessing steps.
Allow users to upload their own CSV files for analysis and prediction.


