import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# Load dataset
file_path = r"C:\Users\shalu\Downloads\The_Cancer_data_1500_V2.csv"
data = pd.read_csv(file_path)

# Handle Missing Values
data = data.dropna()

# Detect and Remove Outliers
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

numerical_columns = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
data = remove_outliers(data, numerical_columns)

# Independent and Target Variables
X = data.drop(columns=['Diagnosis'])  # Independent variables
y = data['Diagnosis']  # Target variable

# Feature Selection using Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(10).index.tolist()
X = X[top_features]

# Add Polynomial Features to Enhance Non-Linear Relationships
poly = PolynomialFeatures(degree=2, include_bias=False)  # Reduced degree for better stability
X_poly = poly.fit_transform(X)

# Power Transformation for better Gaussian-like distribution
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
power_transformer = PowerTransformer()
X_transformed = power_transformer.fit_transform(X_scaled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

# SVR Pipeline
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

# Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(random_state=42)

# Stacking Regressor: Combine SVR and GradientBoosting
stacked_model = StackingRegressor(
    estimators=[('svr', svr_pipeline), ('gb', gb_regressor)],
    final_estimator=RandomForestRegressor(random_state=42)
)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'svr__svr__C': [1, 10, 100, 1000],
    'svr__svr__gamma': [0.001, 0.01, 0.1, 1],
    'svr__svr__epsilon': [0.1, 0.2, 0.3, 0.5],
    'final_estimator__n_estimators': [50, 100, 200],
    'final_estimator__max_depth': [3, 5, 10],
    'final_estimator__min_samples_split': [2, 5, 10]
}

# Grid Search with increased folds (cv=10) for better R² score
grid_search = GridSearchCV(
    estimator=stacked_model,
    param_grid=param_grid,
    cv=10,  # Increased folds
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

# Train the Model
grid_search.fit(X_train, y_train)

# Evaluate the Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Results
print("Best Parameters:", grid_search.best_params_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Save the Model
model_save_path = r"C:\Users\shalu\Downloads\svr_cancer_model_optimized_v3.pkl"
dump(best_model, model_save_path)
print(f"Model saved to {model_save_path}")

# Save the Scaler and Polynomial Features
scaler_save_path = r"C:\Users\shalu\Downloads\scaler.pkl"
poly_save_path = r"C:\Users\shalu\Downloads\poly.pkl"
dump(scaler, scaler_save_path)
dump(poly, poly_save_path)

print(f"Scaler saved to {scaler_save_path}")
print(f"Polynomial Features saved to {poly_save_path}")
