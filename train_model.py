import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the cleaned data
X = pd.read_csv('cleaned_features.csv')
y = pd.read_csv('cleaned_target.csv')

# Make sure y is a 1D array (not DataFrame)
y = y.squeeze()

# Step 1: Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 2: Train the model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Evaluate performance
y_pred = model.predict(X_test)

print("✅ Model training complete!")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 4: Save the model
joblib.dump(model, 'student_dropout_model.joblib')
print("\n💾 Model saved as 'student_dropout_model.joblib'")
