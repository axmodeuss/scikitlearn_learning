from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd


# ============================================================
# 1. Load Dataset
# ============================================================

dataset = fetch_california_housing()

df = pd.DataFrame(
    dataset["data"],
    columns=dataset["feature_names"]
)  # type: ignore

X = df
y = dataset["target"]  # type: ignore


# ============================================================
# 2. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. Baseline Model - Linear Regression
# ============================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)


# ============================================================
# 4. Baseline Error - Mean Prediction
# ============================================================

mean_train = np.mean(y_train)

y_baseline = np.full_like(
    y_test,
    fill_value=mean_train
)

baseline_mae = mean_absolute_error(
    y_test,
    y_baseline
)


# ============================================================
# 5. Decision Tree + Hyperparameter Grid
# ============================================================

base_tree = DecisionTreeRegressor(
    random_state=42
)

param_grid = {
    "max_depth": [3, 5, 10, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10]
}


# ============================================================
# 6. Grid Search + Cross Validation
# ============================================================

grid_search = GridSearchCV(
    estimator=base_tree,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


# ============================================================
# 7. Best Model
# ============================================================

best_params = grid_search.best_params_

best_cv_mae = -grid_search.best_score_

best_model = grid_search.best_estimator_


# ============================================================
# 8. Final Evaluation on Test Set
# ============================================================

test_predictions = best_model.predict(X_test)

final_test_mae = mean_absolute_error(
    y_test,
    test_predictions
)


# ============================================================
# 9. Display Results
# ============================================================

print("\n===== Model Comparison =====")

print("Baseline MAE       :", round(baseline_mae, 4))
print("Linear Regression  :", round(linear_mae, 4))
print("Best CV MAE        :", round(best_cv_mae, 4))
print("Final Test MAE     :", round(final_test_mae, 4))

print("\n===== Best Parameters =====")

for parameter, value in best_params.items():
    print(f"{parameter}: {value}")


# ============================================================
# 10. Grid Search Results
# ============================================================

cv_results = pd.DataFrame(grid_search.cv_results_)

results_table = cv_results[
    [
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "mean_test_score"
    ]
].copy()

results_table["mean_test_score"] = (
    -results_table["mean_test_score"]
)

results_table.columns = [
    "Max Depth",
    "Min Samples Split",
    "Min Samples Leaf",
    "CV MAE"
]

results_table = results_table.sort_values(
    by="CV MAE"
)

print("\n===== Top Grid Search Results =====")

print(
    results_table.head(10).to_string(index=False)
)