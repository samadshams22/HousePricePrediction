
# California House Price Prediction

## Project Description
This project aims to predict house prices in California using the California Housing dataset and the XGBoost Regressor. The project involves data loading, cleaning, visualization, model training, evaluation, and prediction based on user inputs.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Data Loading and Cleaning](#data-loading-and-cleaning)
- [Data Visualization](#data-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [User Interaction](#user-interaction)
- [Prediction](#prediction)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project, you need to have Python installed on your system. The following Python libraries are required:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage
1. Clone this repository to your local machine.
2. Open the Jupyter Notebook `California_House_Price_Prediction.ipynb`.
3. Run the notebook cells to execute the project steps.

## Data Loading and Cleaning
The California Housing dataset is loaded from scikit-learn. The data is checked for missing values and cleaned accordingly. The first few rows of the dataset are displayed to understand its structure.

### Code
```python
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()

data = pd.DataFrame(california.data, columns=california.feature_names)
data['PRICE'] = california.target

print(data.isnull().sum())
print(data.head())
```

## Data Visualization
Visualizations are created to understand the distribution of house prices and the relationships between different features and the target variable. Key visualizations include:
- Distribution of house prices
- Correlation matrix
- Scatter plots for important features

### Code
```python
sns.histplot(data['PRICE'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

correlation_matrix = data.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

features = ['MedInc', 'AveRooms', 'AveOccup']
for feature in features:
    sns.scatterplot(data=data, x=feature, y='PRICE')
    plt.title(f'{feature} vs. PRICE')
    plt.show()
```

## Model Training and Evaluation
The dataset is split into training and testing sets. The XGBoost Regressor is used to train the model. The model's performance is evaluated using the R2 score, and predicted vs actual prices are visualized.

### Code
```python
X = data.drop(columns='PRICE', axis=1)
Y = data['PRICE']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = XGBRegressor()
model.fit(X_train, Y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print(f"Training Data R2 Score: {metrics.r2_score(Y_train, train_predictions)}")
print(f"Testing Data R2 Score: {metrics.r2_score(Y_test, test_predictions)}")

plt.scatter(Y_test, test_predictions)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
plt.title('Predicted vs Actual Prices')
plt.show()
```

## User Interaction
A function is created to prompt the user for input features to predict house prices. This function collects user inputs and organizes them into a DataFrame suitable for prediction.

### Code
```python
def get_user_input():
    print("Enter details for house price prediction:")
    MedInc = float(input("MedInc: "))
    HouseAge = float(input("HouseAge: "))
    AveRooms = float(input("AveRooms: "))
    AveBedrms = float(input("AveBedrms: "))
    Population = float(input("Population: "))
    AveOccup = float(input("AveOccup: "))
    Latitude = float(input("Latitude: "))
    Longitude = float(input("Longitude: "))

    user_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
                             columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    return user_data
```

## Prediction
The user-provided input data is used to predict the house price using the trained model. The predicted price is displayed to the user.

### Code
```python
user_data = get_user_input()
predicted_price = model.predict(user_data)
print(f"The predicted house price is: ${predicted_price[0]:.2f}")
```

## Conclusion
This project demonstrates how to use the California Housing dataset to predict house prices using the XGBoost Regressor. Key steps include data loading, cleaning, visualization, model training, evaluation, and prediction based on user inputs. Future improvements could include adding more features and improving model accuracy.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.
