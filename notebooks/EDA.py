#TASK1- EXPLORATORY DATA ANALYSIS (EDA)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional styling
sns.set_style('whitegrid')

# Load dataset
file_path = 'holiday_dataset.csv'
df = pd.read_csv(r'C:\Users\USER\Downloads\holiday_dataset.csv')

# Display first 5 rows
print(df.head())

# Check dataset shape
print("Dataset Shape:", df.shape)

# Check columns
print("\nColumns:")
print(df.columns)

# Check data types
print("\nData Types:")
print(df.dtypes)

#converting date column to datetime
df['date'] = pd.to_datetime(df['date'])
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:", df.shape)

print(df.describe(include='all'))
print(df.mode().iloc[0])

# Extract year and month
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Extract month name
df['month_name'] = df['date'].dt.month_name()

# Count events by year
events_per_year = df.groupby('year').size()
print(events_per_year)

# Select numeric columns only
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Mean
print("Mean:")
print(numeric_columns.mean())

# Median
print("\nMedian:")
print(numeric_columns.median())

# Mode
print("\nMode:")
print(df.mode().iloc[0])

plt.figure(figsize=(10,5))

plt.plot(events_per_year.index,
         events_per_year.values,
         marker='o')

plt.title('Holiday Events Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Events')
plt.show()

monthly_events = df.groupby('month_name').size()

# Reorder months
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

monthly_events = monthly_events.reindex(month_order)
print(monthly_events)

plt.figure(figsize=(12,6))

sns.barplot(x=monthly_events.index,
            y=monthly_events.values)

plt.xticks(rotation=45)
plt.title('Holiday Events by Month')
plt.xlabel('Month')
plt.ylabel('Number of Events')
plt.show()

holiday_types = df['type'].value_counts()

print(holiday_types)

plt.figure(figsize=(8,5))

sns.countplot(data=df,
              x='type',
              order=df['type'].value_counts().index)

plt.title('Distribution of Holiday Types')
plt.xlabel('Holiday Type')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.show()

print(df['transferred'].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(data=df, x='transferred')

plt.title('Transferred Holidays Distribution')
plt.xlabel('Transferred')
plt.ylabel('Count')

plt.show()