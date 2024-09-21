
3-Disease Prediction from Medical Data


data preprocessing 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Preview the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Feature correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Split features and labels
X = df.drop('Outcome', axis=1)  # Features (medical data)
y = df['Outcome']  # Target (disease)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model building 


from sklearn.ensemble import RandomForestClassifier

# Build the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)


model evaluation 


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))


prediction on new data 


# Example of a new patient record (glucose, blood pressure, etc.)
new_patient_data = np.array([[120, 80, 35, 25, 150, 33.1, 0.627, 50]])

# Scale the new data
new_patient_data = scaler.transform(new_patient_data)

# Predict the outcome (disease likelihood)
prediction = model.predict(new_patient_data)

# Output the prediction (1 = Disease, 0 = No Disease)
if prediction[0] == 1:
    print("The model predicts that the patient is likely to have the disease.")
else:
    print("The model predicts that the patient is unlikely to have the disease.")


alternative models and further improvements 


from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Evaluate the Logistic Regression model
logreg_pred = logreg_model.predict(X_test)
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, logreg_pred) * 100:.2f}%')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build a Neural Network model
nn_model = Sequential()
nn_model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the Neural Network
nn_test_loss, nn_test_acc = nn_model.evaluate(X_test, y_test)
print(f'Neural Network Test Accuracy: {nn_test_acc * 100:.2f}%')

