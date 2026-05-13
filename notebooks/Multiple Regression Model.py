#Multiple Regression Model
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import numpy as np

sales_df = pd.read_csv('data/modified_sales_dataset.csv')

stores_df = pd.read_csv('data/stores.csv')
print(sales_df.columns)
print(stores_df.columns)
#assigning store number to dataset
np.random.seed(42)

sales_df['store_nbr'] = np.random.randint(
    1,
    len(stores_df) + 1,
    size=len(sales_df)
)
#merging dataset
df = pd.merge(sales_df, stores_df, on='store_nbr')

print(sales_df.head())
print(stores_df.head())

df['type_y'] = df['type_y'].astype('category').cat.codes

df['city'] = df['city'].astype('category').cat.codes
#check missing value
print(df.isnull().sum())
#remove empty roles
df = df.dropna()
#using multiple variables
X = df[['promotion', 'holiday', 'type_y', 'cluster']]
y = df['sales']
#splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#train model
model = LinearRegression()
model.fit(X_train, y_train)
#make prediction
predictions = model.predict(X_test)
print(predictions[:5])
#evaluate the model
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)