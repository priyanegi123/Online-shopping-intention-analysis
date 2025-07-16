# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the dataset
data = pd.read_csv('C:/Users/priya/Downloads/online_shoppers_intention-_1_.csv')  # Update the path if needed

# Step 3: Data Overview
print("Dataset loaded successfully!")
print(data.head())  # Display first 5 rows

# Step 4: Handle missing values (if any)
print("Missing values in the dataset:")
print(data.isnull().sum())
data = data.dropna()  # Drop rows with missing values

# Step 5: Exploratory Data Analysis (EDA)
print("Performing EDA...")

# Filter only numeric columns for correlation
numeric_data = data.select_dtypes(include=[np.number])  # Select numeric columns only

# Reduce the size of the heatmap to show a manageable number of features
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()  # Show the heatmap one by one

# Revenue distribution (simpler version)
sns.countplot(x='Revenue', data=data)
plt.title('Revenue Distribution')
plt.xlabel('Revenue (0 = No Purchase, 1 = Purchase)')
plt.ylabel('Count')
plt.show()
  # Explicitly show this plot

# Month vs Revenue
plt.figure(figsize=(6, 4))
sns.countplot(x='Month', hue='Revenue', data=data)
plt.title("Revenue by Month")
plt.show()  # Explicitly show this plot

# Step 6: Encode categorical variables
categorical_columns = ['Month', 'VisitorType', 'Weekend']
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Step 7: Split the data into features (X) and target (y)
X = data.drop('Revenue', axis=1)  # 'Revenue' is the target variable
y = data['Revenue']

# Step 8: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 9: Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Train Logistic Regression Model with improved settings
model_lr = LogisticRegression(max_iter=500, solver='saga')
model_lr.fit(X_train, y_train)

# Step 11: Make predictions with Logistic Regression
y_pred_lr = model_lr.predict(X_test)

# Step 12: Evaluate Logistic Regression Model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix for Logistic Regression
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

# Step 13: Train Random Forest Model
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Step 14: Make predictions with Random Forest
y_pred_rf = model_rf.predict(X_test)

# Step 15: Evaluate Random Forest Model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Random Forest)")
plt.show()

# Step 16: Feature Importance from Random Forest
feature_importances = model_rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# Step 17: Plot Feature Importances
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importances (Random Forest)")
plt.show()

# Step 18: Cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(model_lr, X, y, cv=5)  # 5-fold cross-validation
print(f"Logistic Regression Cross-validated accuracy: {cv_scores_lr.mean() * 100:.2f}%")

# Step 19: Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
grid_search_rf.fit(X_train, y_train)
print(f"Best parameters found: {grid_search_rf.best_params_}")

# Step 20: Evaluate the tuned Random Forest model
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf_tuned = best_rf_model.predict(X_test)
accuracy_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
print(f"Tuned Random Forest Accuracy: {accuracy_rf_tuned * 100:.2f}%")
print("Classification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_rf_tuned))