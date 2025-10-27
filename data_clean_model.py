import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap
import matplotlib.pyplot as plt

# Load raw dataset
df = pd.read_csv('Dataset.csv')

# Basic numeric imputation
num_cols = df.select_dtypes(include='number').columns
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

# Flags creation for risk factors
df['poor_attendance_flag'] = (df['attendance_rate'] < 70).astype(int)
df['academic_stress_flag'] = (df['stress_level'] > 7).astype(int)
df['low_support_flag'] = (df['mentor_meetings'] < 2).astype(int)
df['financial_issues_flag'] = (df['financial_status'] == 'Low').astype(int)

# Categorical encoding
cat_cols = ['gender', 'department', 'scholarship', 'parental_education', 'extra_curricular', 'sports_participation']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_cols))

# Combine all features
df_final = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

# Target and features
X = df_final.drop('dropout', axis=1)
y = df_final['dropout']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42)),
])

pipeline.fit(X_train, y_train)

# Save model and feature order
joblib.dump(pipeline, 'model_pipeline.pkl')
joblib.dump(list(X.columns), 'feature_order.pkl')

# SHAP explainability
explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
shap_values = explainer.shap_values(X_train)
plt.figure()
shap.summary_plot(shap_values[1], X_train, show=False)
plt.savefig('shap_summary.png')
