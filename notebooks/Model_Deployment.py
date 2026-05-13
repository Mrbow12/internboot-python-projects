#Model Deployment (FlaskDashboard)
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('../data/train.csv')

# View columns
print(df.columns)

# Remove missing values
df = df.dropna()

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Extract month and day
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Features
X = df[['store_nbr', 'family', 'onpromotion', 'month', 'day']]

# Convert family column to numbers
X = pd.get_dummies(X, columns=['family'])

# Target
y = df['sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('sales_model.pkl', 'wb'))

# Save training columns
pickle.dump(X.columns, open('model_columns.pkl', 'wb'))

print("Model trained successfully")