# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

# AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

# Program:
```
# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load the Dataset (upload the file in Colab!)
df = pd.read_csv('Placement_Data-3.csv')

# Step 3: Quick Data Inspection
print(df.head())
print(df.info())

# Step 4: Preprocessing
# Drop any irrelevant columns if needed (e.g., serial number, salary after placement)
df = df.drop(['slno', 'salary'], axis=1, errors='ignore')

# Encode categorical features
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Step 5: Splitting Data
X = df.drop('status', axis=1)
y = df['status']  # 'status' is Placed/Not Placed

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Training the Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Step 8: Predict & Evaluate
y_pred = log_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# (Optional) Predict for a single student (just as example)
# sample = np.array([[1, 67, 2, 91, 2, 0, 58, 1, 0, 55, 1, 0, 0, 58.8]]) # this needs to match input order and encoding
# print('Predicted Placement Status:', log_model.predict(sample))


```

# Output:

<img width="958" height="822" alt="image" src="https://github.com/user-attachments/assets/2bbcd52b-7638-4a69-8882-153b33fc53ce" />

<img width="681" height="336" alt="image" src="https://github.com/user-attachments/assets/6eb580fa-a032-470e-9608-1a87b9b8e7ce" />

# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
