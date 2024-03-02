import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset (you need to download and provide your own dataset)
# For the purpose of this example, I'll use a placeholder dataset
# Replace 'your_dataset.csv' with the actual path to your dataset
# The dataset should have features (independent variables) and a target variable indicating Alzheimer's status.

# df = pd.read_csv('your_dataset.csv')

# For this example, let's create a placeholder dataset
# You should replace this with your actual dataset


data = {
    'Age': [70, 65, 80, 75, 60],
    'Memory_Score': [0.8, 0.7, 0.5, 0.2, 0.6],
    'Brain_Size': [1200, 1100, 900, 800, 1000],
    'Alzheimer_Status': [1, 0, 1, 1, 0]  # 1: Alzheimer's, 0: No Alzheimer's
}


df = pd.DataFrame(data)

# Split the dataset into features and target variable
X = df.drop('Alzheimer_Status', axis=1)
y = df['Alzheimer_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier model (you can experiment with other algorithms)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

pickle.dump(model,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
