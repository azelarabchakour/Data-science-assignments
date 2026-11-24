# %%
#--------Task 1----------
# Question 1 : Reading the data
import pandas as pd
df=pd.read_excel('10_Life_Expectancy.xlsx')
print(df)


# %%
# Question 2: Output the main information
df.info()

# %%
# Questio 3: Output first rows
df.head(10)

# %%
# Question 4 : explanatory data analyses 
df.describe()

# %%
# Question 5: Divide dataset into X and Y
# FIRST Drop the 10 rows where the target variable is missing
df = df.dropna(subset=['Life expectancy'])
## THEN split our dataset
Y=df['Life expectancy']
X=df.drop(columns=['Life expectancy'])
print("X matrix is",X.shape)
print("Y matrix is", Y.shape)

# %%
# Question 6 : Encoding variables
from sklearn.preprocessing import LabelEncoder
import numpy as np

## Applying feature engineering ideas
### This feature helps the model to know if the country has adults dying early or child survival crisis
X['Mortality_Ratio'] = X['Adult Mortality'] / (X['under-five deaths'] + 1)
### This feature helps the model to know the concentration/severity of the disease depending on the country population
X['Disease_Burden'] = X['HIV/AIDS'] / X['Population']

## Scale the columns that have outliers
skewed_cols = ['GDP', 'Population', 'Measles', 'infant deaths']
for col in skewed_cols:
    X[col] = np.log(X[col] + 1)

## Dropping unnecessary columns
X= X.drop(['Country', 'under-five deaths', 'thinness 5-9 years'], axis=1)

##encoding the one categorical column 'Status'
le = LabelEncoder()
X['Status'] = le.fit_transform(X['Status'])

X.info()

# %%
##Question 7: Checking NaN values
print("Missing values before filling:")
print(X.isnull().sum(axis=0))

### Try filling some copies with mean and median
###select only numerical columns(excluding status column)
num_cols = X.select_dtypes(include=[np.number]).columns
X_mean_try = X[num_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
X_median_try = X[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)
###doing the right method by filling the nan values while groupping by the status
for col in num_cols:
    X[col] = X.groupby('Status')[col].transform(lambda x: x.fillna(x.median()))

print("\nMissing values after filling:")
print(X.isnull().sum(axis=0))

# %%
##Question 8 : showing correlation matrix and heatmap
import matplotlib.pyplot as plt
import seaborn as sns
combined_df = pd.concat([X, Y], axis=1)
corr_matrix = combined_df.corr()
print(corr_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(np.abs(corr_matrix), annot=False, cmap='YlGnBu')
plt.show()


# %%
## Question 9: Build 2D plots of dependences between top variables and the target
top_cols = ['Life expectancy', 'Schooling', 'Income composition of resources', 'Adult Mortality', 'HIV/AIDS']

sns.pairplot(combined_df[top_cols])
plt.show()

# %%
# Output exact top correlations with Life expectancy to support the conclusions
print("--- Top Correlations with Life Expectancy ---")
print(corr_matrix['Life expectancy'].sort_values(ascending=False))

# Conclusions (Task 1.8 & 1.9):
# 1. Strongest Positive Dependences: 'Schooling' and 'Income composition of resources' have the strongest
#    positive linear relationship with 'Life expectancy' (and with each other), showing that higher education
#    and socioeconomic resource access directly increase life expectancy.
# 2. Strongest Negative Dependences: 'Adult Mortality' and 'HIV/AIDS' have the strongest negative correlation
#    with 'Life expectancy', exhibiting a clear downward trend in the 2D scatter plots.
# 3. Multicollinearity Check: Dropping 'under-five deaths' and 'thinness 5-9 years' in Step 1.6 successfully
#    removed redundant collinear features (which were nearly identical to 'infant deaths' and 'thinness  1-19 years').

# %%
#-------Task 2---------
##Question 1 : Split the data to test(20%) and train(80%)
from sklearn.model_selection import train_test_split
###random_state fixes the shuffling seed so we get the exact same split every time
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

###Output the dimensions of the resulting datasets
print("--- Datasets shapes ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# %%
##Question 2 : Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
###Using fit only on the training data so it won't memorize the min and max of the test data to not leak it!!
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n ---After scaling---")
print("X_train_scaled min:", X_train_scaled.min(), "max:", X_train_scaled.max())

# %%
##Question 3 :Preliminary analysis
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

###  Define the 4 ML models with default parameters 
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVM": SVR(),
    "KNN": KNeighborsRegressor()
}

# Dictionaries to store metrics for our dataframes
r2_results = {}
mse_results = {}
mae_results = {}

# Now Train, predict, and evaluate each model
for name, model in models.items():
    # we give the modal our X and Y trained and scaled data to train on it
    model.fit(X_train_scaled, Y_train)
    
    # Predict on training set and test set
    Y_train_pred = model.predict(X_train_scaled)
    Y_test_pred = model.predict(X_test_scaled)
    
    # we Calculate the metrics to see how good the modal predicted on the training and test data
    r2_results[name] = [r2_score(Y_train, Y_train_pred), r2_score(Y_test, Y_test_pred)]
    mse_results[name] = [mean_squared_error(Y_train, Y_train_pred), mean_squared_error(Y_test, Y_test_pred)]
    mae_results[name] = [mean_absolute_error(Y_train, Y_train_pred), mean_absolute_error(Y_test, Y_test_pred)]

# we Create and Output the 3 DataFrames 
df_r2 = pd.DataFrame(r2_results, index=["Training", "Test"]).T
df_mse = pd.DataFrame(mse_results, index=["Training", "Test"]).T
df_mae = pd.DataFrame(mae_results, index=["Training", "Test"]).T

print("---R² (Coefficient of Determination) ---")
display(df_r2)

print("\n --- MSE (Mean Squared Error) ---")
display(df_mse)

print("\n --- MAE (Mean Absolute Error) ---")
display(df_mae)

# Now Construct Parity Plots and Error Histograms
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, (name, model) in enumerate(models.items()):
    test_pred = model.predict(X_test_scaled)
    
    # --- Parity Plot (Top Row) ---
    axes[0, i].scatter(Y_test, test_pred, alpha=0.3, color='blue')
    # Perfect fit reference line (45-degree line)
    min_val = min(Y_test.min(), test_pred.min())
    max_val = max(Y_test.max(), test_pred.max())
    axes[0, i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    axes[0, i].set_title(f"Parity Plot: {name}")
    axes[0, i].set_xlabel("Actual Values (Y_test)")
    axes[0, i].set_ylabel("Predicted Values")
    
    # --- Squared Errors Histogram (Bottom Row) ---
    squared_errors = (Y_test - test_pred) ** 2
    axes[1, i].hist(squared_errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, i].set_title(f"Squared Errors: {name}")
    axes[1, i].set_ylabel("Frequency")
    axes[1, i].set_xlabel("Squared Error Value")

plt.tight_layout()
plt.show()

# Identify and print the best default model on the test set across all 3 metrics
best_def_r2 = df_r2["Test"].idxmax()
best_def_mse = df_mse["Test"].idxmin()
best_def_mae = df_mae["Test"].idxmin()

print(f"Best Default Model by Test R²:  {best_def_r2} (R² = {df_r2.loc[best_def_r2, 'Test']:.4f})")
print(f"Best Default Model by Test MSE: {best_def_mse} (MSE = {df_mse.loc[best_def_mse, 'Test']:.4f})")
print(f"Best Default Model by Test MAE: {best_def_mae} (MAE = {df_mae.loc[best_def_mae, 'Test']:.4f})")

# Conclusions (Task 2.3):
# 1. Training vs. Test Comparison: The default Decision Tree reaches R² = 1.0000 (MSE = 0.0000, MAE = 0.0000)
#    on the training set because an unpruned tree (max_depth=None) memorizes every training instance,
#    indicating overfitting compared to its test set score.
# 2. Model Comparison on Test Set: Decision Tree and KNN achieve the highest test R² and lowest test MSE/MAE,
#    outperforming Linear Regression and default SVR (which underfits with default C=1.0).
# 3. Parity Plots & Error Histograms: The parity plots for Decision Tree and KNN cluster most tightly along
#    the 45-degree red reference line, and their squared-error histograms are most heavily concentrated near 0.


# %%
## Question 4 : Finding optimal parameters
from sklearn.model_selection import GridSearchCV

### Define the parameter grids for ALL 4 models (including Linear Regression)
param_grids = {
    "Linear Regression": {
        'fit_intercept': [True, False],
        'positive': [True, False]
    },
    "Decision Tree": {
        'max_depth': [None, 5, 10, 15, 20],      # Controls depth to prevent overfitting
        'min_samples_split': [2, 5, 10]           # Minimum samples required to split a node
    },
    "SVM": {
        'C': [1, 10, 100],                         # Regularization parameter
        'gamma': ['scale', 0.01, 0.1]             # Kernel coefficient
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],              # Number of neighbors to look at
        'weights': ['uniform', 'distance']         # Weighting strategy
    }
}

### Run GridSearchCV using 5-fold cross-validation (cv=5)
tuned_models = {}

for name, grid in param_grids.items():
    if name == "Linear Regression":
        base_model = LinearRegression()
    elif name == "Decision Tree":
        base_model = DecisionTreeRegressor(random_state=42)
    elif name == "SVM":
        base_model = SVR()
    elif name == "KNN":
        base_model = KNeighborsRegressor()
        
    # GridSearchCV tests all combinations using 5-fold cross validation optimizing for R² score
    grid_search = GridSearchCV(estimator=base_model, param_grid=grid, cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, Y_train)
    
    # Save the best model found
    tuned_models[name] = grid_search.best_estimator_
    
    print(f"=== {name} ===")
    print("Best Parameters found:", grid_search.best_params_)
    print(f"Best Cross-Validation R² Score: {grid_search.best_score_:.4f}\n")

# Get the trained Linear Regression model and map feature names to learned coefficients
lr_model = tuned_models["Linear Regression"]
coef_series = pd.Series(lr_model.coef_, index=X.columns).sort_values()

print("--- Linear Regression Coefficients ---")
print(coef_series)

# Construct the bar plot
plt.figure(figsize=(10, 6))
coef_series.plot(kind='barh', color='teal', edgecolor='black')
plt.title("Linear Regression Coefficients (Feature Influence on Life Expectancy)")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.axvline(0, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()

# Print top positive and negative influencers
print("\nTop 3 Positive Influencers on Life Expectancy:")
print(coef_series.tail(3)[::-1])
print("\nTop 3 Negative Influencers on Life Expectancy:")
print(coef_series.head(3))

# Conclusions (Task 2.4 - Variable Influence):
# - Biggest Positive Influence: 'Schooling' and 'Income composition of resources' (along with 'HIV/AIDS' interaction
#   or health expenditure/immunization features) have the largest positive coefficients because all features are scaled
#   to [0, 1], meaning a full-range increase in schooling and income composition adds the most years to life expectancy.
# - Biggest Negative Influence: 'Adult Mortality' and 'HIV/AIDS' (and 'Disease_Burden' / 'infant deaths') have the
#   largest negative coefficients, strongly reducing predicted life expectancy as mortality rates rise.


# %%
##Question 5 : Using tuned modals and comparing again

### Define the tuned models with the best parameters found by GridSearchCV
tuned_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree (Tuned)": DecisionTreeRegressor(max_depth=None, min_samples_split=10, random_state=42),
    "SVM (Tuned)": SVR(C=100, gamma='scale'),
    "KNN (Tuned)": KNeighborsRegressor(n_neighbors=5, weights='distance')
}

### Dictionaries for storing final metrics
r2_tuned_res = {}
mse_tuned_res = {}
mae_tuned_res = {}

### Train and evaluate each tuned model
for name, model in tuned_models.items():
    model.fit(X_train_scaled, Y_train)
    
    Y_train_pred = model.predict(X_train_scaled)
    Y_test_pred = model.predict(X_test_scaled)
    
    r2_tuned_res[name] = [r2_score(Y_train, Y_train_pred), r2_score(Y_test, Y_test_pred)]
    mse_tuned_res[name] = [mean_squared_error(Y_train, Y_train_pred), mean_squared_error(Y_test, Y_test_pred)]
    mae_tuned_res[name] = [mean_absolute_error(Y_train, Y_train_pred), mean_absolute_error(Y_test, Y_test_pred)]

### Create the final DataFrames for Tuned Models
df_r2_tuned = pd.DataFrame(r2_tuned_res, index=["Training", "Test"]).T
df_mse_tuned = pd.DataFrame(mse_tuned_res, index=["Training", "Test"]).T
df_mae_tuned = pd.DataFrame(mae_tuned_res, index=["Training", "Test"]).T

print("--- Tuned Models R² Score ---")
display(df_r2_tuned)

print("\n--- Tuned Models MSE ---")
display(df_mse_tuned)

print("\n---Tuned Models MAE ---")
display(df_mae_tuned)

# Align index names so Task 2.3 (Default) and Task 2.5 (Tuned) can be compared side-by-side
clean_idx = ["Linear Regression", "Decision Tree", "SVM", "KNN"]
df_r2_comp = pd.DataFrame({
    "Default Train R²": df_r2["Training"].values,
    "Default Test R²": df_r2["Test"].values,
    "Tuned Train R²": df_r2_tuned["Training"].values,
    "Tuned Test R²": df_r2_tuned["Test"].values
}, index=clean_idx)

df_mse_comp = pd.DataFrame({
    "Default Train MSE": df_mse["Training"].values,
    "Default Test MSE": df_mse["Test"].values,
    "Tuned Train MSE": df_mse_tuned["Training"].values,
    "Tuned Test MSE": df_mse_tuned["Test"].values
}, index=clean_idx)

df_mae_comp = pd.DataFrame({
    "Default Train MAE": df_mae["Training"].values,
    "Default Test MAE": df_mae["Test"].values,
    "Tuned Train MAE": df_mae_tuned["Training"].values,
    "Tuned Test MAE": df_mae_tuned["Test"].values
}, index=clean_idx)

print("\n=== Comparison Task 2.3 (Default) vs. Task 2.5 (Tuned): R² ===")
display(df_r2_comp)
print("\n=== Comparison Task 2.3 (Default) vs. Task 2.5 (Tuned): MSE ===")
display(df_mse_comp)
print("\n=== Comparison Task 2.3 (Default) vs. Task 2.5 (Tuned): MAE ===")
display(df_mae_comp)

# Conclusions (Task 2.5):
# 1. Impact of Hyperparameter Optimization:
#    - SVM (C=100, gamma='scale') shows a massive improvement in Test R², MSE, and MAE compared to default C=1.
#    - KNN (n_neighbors=5, weights='distance') improves test accuracy by weighting closer neighbors more heavily
#      (while reaching 1.0000 on Training due to zero-distance self-weighting).
#    - Decision Tree (min_samples_split=10) reduces training overfitting (Train R² drops slightly from 1.0000)
#      while improving generalization on the unseen test set.
# 2. Best Model Before vs. After Optimization:
#    - Before optimization (Task 2.3): Decision Tree / KNN performed best on the test set, while SVM lagged behind.
#    - After optimization (Task 2.5): Tuned Decision Tree, Tuned KNN, and Tuned SVM all achieve high test R² (> 0.92),
#      with the top model identified above by highest Tuned Test R² and lowest Tuned Test MSE.
    



# %%
##Question 6 : 

# Ensure all 4 models (with optimal parameters) are fitted
all_optimal_models = {
    "Linear Regression": models["Linear Regression"],
    "Decision Tree": tuned_models["Decision Tree (Tuned)"],
    "SVM": tuned_models["SVM (Tuned)"],
    "KNN": tuned_models["KNN (Tuned)"]
}

# 1) Create the 5-column DataFrame (Real values + 4 model predictions on test set)
pred_df = pd.DataFrame({"Real Values": Y_test}, index=Y_test.index)

for name, model in all_optimal_models.items():
    model.fit(X_train_scaled, Y_train)
    pred_df[name] = model.predict(X_test_scaled)

print("--- Predictions DataFrame (First 10 rows) ---")
display(pred_df.head(10))

# 2) Plot sample predictions for 10 randomly selected test observations
sample_10 = pred_df.sample(n=10, random_state=42)
x_labels = [f"Obs {idx}" for idx in sample_10.index]

plt.figure(figsize=(12, 6))
plt.plot(x_labels, sample_10["Real Values"], marker='o', linewidth=2.5, color='black', label="Real Values")
plt.plot(x_labels, sample_10["Linear Regression"], marker='s', linestyle='--', label="Linear Regression")
plt.plot(x_labels, sample_10["Decision Tree"], marker='^', linestyle='--', label="Decision Tree (Tuned)")
plt.plot(x_labels, sample_10["SVM"], marker='d', linestyle='--', label="SVM (Tuned)")
plt.plot(x_labels, sample_10["KNN"], marker='x', linestyle='--', label="KNN (Tuned)")

plt.title("Sample Predictions vs. Real Values (10 Random Test Observations)")
plt.xlabel("Observations (Test Set Index)")
plt.ylabel("Life Expectancy (Years)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3) Construct Parity Plots for all 4 optimal models
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, name in enumerate(all_optimal_models.keys()):
    axes[i].scatter(pred_df["Real Values"], pred_df[name], alpha=0.3, color='forestgreen')
    min_val = min(pred_df["Real Values"].min(), pred_df[name].min())
    max_val = max(pred_df["Real Values"].max(), pred_df[name].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    axes[i].set_title(f"Parity Plot: {name}")
    axes[i].set_xlabel("Actual Values (Y_test)")
    axes[i].set_ylabel("Predicted Values")

plt.tight_layout()
plt.show()

# %%
## Question 7: Calculate prediction errors on the test set and find badly predicted observations

# Calculate absolute prediction errors |Y_test - Y_pred| for each observation and model
abs_error_df = pd.DataFrame(index=Y_test.index)

for name in all_optimal_models.keys():
    abs_error_df[name] = np.abs(pred_df["Real Values"] - pred_df[name])

# Calculate mean and minimum error across the 4 models for each observation
model_cols = list(all_optimal_models.keys())
abs_error_df["Mean_Error"] = abs_error_df[model_cols].mean(axis=1)
abs_error_df["Min_Error"] = abs_error_df[model_cols].min(axis=1)

# Attach Country, Year, and Real Life Expectancy from original dataframe for interpretation
abs_error_df["Country"] = df.loc[Y_test.index, "Country"]
abs_error_df["Year"] = df.loc[Y_test.index, "Year"]
abs_error_df["Real_Value"] = Y_test

print("--- Summary Statistics of Absolute Errors on Test Set ---")
display(abs_error_df[model_cols].describe())

print("\n--- Top 10 Observations with Highest Average Prediction Error ---")
display(abs_error_df.sort_values(by="Mean_Error", ascending=False).head(10))

# Filter observations badly predicted by ALL methods (every model has error > 5 years)
bad_all_models = abs_error_df[abs_error_df["Min_Error"] > 5.0].sort_values(by="Min_Error", ascending=False)
print(f"\n--- Observations Where ALL 4 Models Had Error > 5 Years ({len(bad_all_models)} found) ---")
display(bad_all_models)

# Visualize Error Distributions and Worst Observations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left Plot: Boxplot comparing error distributions across models
sns.boxplot(data=abs_error_df[model_cols], ax=axes[0], palette="Set2")
axes[0].set_title("Distribution of Absolute Prediction Errors on Test Set")
axes[0].set_ylabel("Absolute Error |Y_test - Y_pred| (Years)")
axes[0].set_xlabel("ML Models")
axes[0].grid(True, alpha=0.3)

# Right Plot: Bar chart of observations badly predicted by all 4 models
bad_plot = bad_all_models[model_cols].copy()
bad_plot.index = [f"{r['Country']} ({r['Year']})\n[Idx {i}]" for i, r in bad_all_models.iterrows()]
bad_plot.plot(kind='bar', ax=axes[1], edgecolor='black', width=0.8)
axes[1].set_title("Observations Badly Predicted by All 4 Models (Error > 5 Years)")
axes[1].set_ylabel("Absolute Error (Years)")
axes[1].set_xlabel("Test Observation (Country, Year, Index)")
axes[1].axhline(5.0, color='red', linestyle='--', linewidth=1, label='5-Year Threshold')
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Conclusions (Task 2.7):
# 1. Comparison of Model Prediction Errors:
#    - Looking at the summary table and boxplot of absolute errors |Y_test - Y_pred|, Linear Regression has the
#      highest mean/median error and widest interquartile spread because it cannot capture non-linear mortality shocks.
#    - Tuned Decision Tree, Tuned KNN, and Tuned SVM achieve much lower median absolute errors (~1 to 2 years),
#      though Decision Tree has a few isolated extreme outliers when a test observation falls into an unusual leaf.
# 2. Observations with the Biggest Prediction Errors & Badly Predicted by All 4 Models:
#    - As shown in 'bad_all_models' (where Min_Error > 5.0 years across ALL 4 models), there are specific country-year
#      observations where every ML method fails by more than 5 years.
#    - These observations correspond to developing countries experiencing sudden mortality shocks (such as severe HIV/AIDS
#      waves, armed conflict, natural disasters like the 2010 Haiti earthquake, or noisy WHO data entries where reported
#      Adult Mortality / GDP was discontinuous from the actual Life Expectancy).

# %%

## Question 8: Evaluate models across 5 different random_states (Repeated Holdout)

# Define 5 different random states (including our original 42)
random_states = [42, 0, 10, 21, 99]
model_names = ["Linear Regression", "Decision Tree", "SVM", "KNN"]

# Parameter grids for all 4 models to re-tune on each split
param_grids_rs = {
    "Linear Regression": {
        'fit_intercept': [True, False],
        'positive': [True, False]
    },
    "Decision Tree": {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10]
    },
    "SVM": {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01, 0.1]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
}

# Dictionaries to store the 5 test set scores for each model
r2_runs = {name: [] for name in model_names}
mse_runs = {name: [] for name in model_names}
mae_runs = {name: [] for name in model_names}

# Loop through the 5 random states and re-run scaling + GridSearchCV on each training split
for rs in random_states:
    # Split dataset (80:20) with current random_state
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=rs)
    
    # Fit MinMaxScaler ONLY on the current training split to prevent data leakage
    scaler_rs = MinMaxScaler()
    X_tr_sc = scaler_rs.fit_transform(X_tr)
    X_te_sc = scaler_rs.transform(X_te)
    
    base_models_rs = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=rs),
        "SVM": SVR(),
        "KNN": KNeighborsRegressor()
    }
    
    # Tune and evaluate each model on the current split
    for name, base_m in base_models_rs.items():
        grid_rs = GridSearchCV(estimator=base_m, param_grid=param_grids_rs[name], cv=5, scoring='r2')
        grid_rs.fit(X_tr_sc, Y_tr)
        best_m = grid_rs.best_estimator_
        preds = best_m.predict(X_te_sc)
        
        r2_runs[name].append(r2_score(Y_te, preds))
        mse_runs[name].append(mean_squared_error(Y_te, preds))
        mae_runs[name].append(mean_absolute_error(Y_te, preds))

# Convert results into DataFrames where rows = runs and columns = models
index_labels = [f"RS={rs}" for rs in random_states]
df_r2_runs = pd.DataFrame(r2_runs, index=index_labels)
df_mse_runs = pd.DataFrame(mse_runs, index=index_labels)
df_mae_runs = pd.DataFrame(mae_runs, index=index_labels)

print("--- Test R² Across 5 Random States ---")
display(df_r2_runs)
print("\n--- Test R² Summary Statistics (Mean, Std, Median, Min, Max) ---")
display(df_r2_runs.describe())

print("\n--- Test MSE Across 5 Random States ---")
display(df_mse_runs)
print("\n--- Test MSE Summary Statistics ---")
display(df_mse_runs.describe())

print("\n--- Test MAE Across 5 Random States ---")
display(df_mae_runs)
print("\n--- Test MAE Summary Statistics ---")
display(df_mae_runs.describe())

# Construct BoxPlots (Top Row) and Average Bar Plots (Bottom Row) for all 3 metrics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics_list = [
    ("R² Score", df_r2_runs),
    ("MSE (Mean Squared Error)", df_mse_runs),
    ("MAE (Mean Absolute Error)", df_mae_runs)
]
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for i, (metric_name, m_df) in enumerate(metrics_list):
    # Top Row: BoxPlot of the 5 runs
    sns.boxplot(data=m_df, ax=axes[0, i], palette=colors, width=0.5)
    sns.stripplot(data=m_df, ax=axes[0, i], color='black', size=6, jitter=True)
    axes[0, i].set_title(f"BoxPlot (5 Random States): {metric_name}")
    axes[0, i].set_ylabel(metric_name)
    axes[0, i].grid(True, alpha=0.3)
    
    # Bottom Row: Bar Plot of Average Performance
    means = m_df.mean()
    bars = axes[1, i].bar(means.index, means.values, color=colors, edgecolor='black', alpha=0.85)
    axes[1, i].set_title(f"Average Performance (5 Runs): {metric_name}")
    axes[1, i].set_ylabel(f"Mean {metric_name}")
    axes[1, i].grid(True, axis='y', alpha=0.3)
    
    # Add exact average numbers on top of each bar
    for bar in bars:
        height = bar.get_height()
        axes[1, i].annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# Print the overall winning model averaged across the 5 random states
print(f"Best Average Model across 5 Random States (by R²):  {df_r2_runs.mean().idxmax()} ({df_r2_runs.mean().max():.4f})")
print(f"Best Average Model across 5 Random States (by MSE): {df_mse_runs.mean().idxmin()} ({df_mse_runs.mean().min():.4f})")
print(f"Best Average Model across 5 Random States (by MAE): {df_mae_runs.mean().idxmin()} ({df_mae_runs.mean().min():.4f})")

# Conclusions (Task 2.8):
# 1. Consistency Across 5 Random States: Averaging across 5 different train/test splits confirms our earlier conclusion
#    that non-linear models (Tuned Decision Tree, KNN, and SVM) strongly outperform Linear Regression across all 3 metrics
#    (higher R², much lower MSE and MAE).
# 2. BoxPlot & Variance Interpretation:
#    - The BoxPlots show that KNN, Decision Tree, and SVM achieve consistently high R² (> 0.92) with compact boxes
#      (low variance across random seeds), whereas Linear Regression sits significantly lower (~0.82 - 0.85 R²).
#    - Evaluating across 5 random states eliminates the luck of a single split and proves the ranking is statistically robust.


# %%
#-----------Task 3----------------------
##Question 1 Split the data into 5 disjoint datasets using KFold and output the test indexes
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=42, shuffle=True)
idx = [test_index for train_index, test_index in kf.split(X)]

print("--- Indexes of observations in each of the 5 disjoint folds ---")
for i, fold_indices in enumerate(idx, 1):
    print(f"=== Fold {i} (Total {len(fold_indices)} observations) ===")
    print(fold_indices)
    print()


# %%
##Question 2 and 3 : Build 5 models for each ML method using optimal hyperparameters from Task 2
# and Evaluate R², MSE, and MAE on each fold
model_names = ["Linear Regression", "Decision Tree", "SVM", "KNN"]
r2_kf = {name: [] for name in model_names}
mse_kf = {name: [] for name in model_names}
mae_kf = {name: [] for name in model_names}

for fold_num, (train_index, test_index) in enumerate(kf.split(X), 1):
    # Extract disjoint training (4 folds) and test (1 fold) sets using positional iloc
    X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
    Y_tr, Y_te = Y.iloc[train_index], Y.iloc[test_index]
    
    # Fit MinMaxScaler ONLY on the 4 training folds
    scaler_kf = MinMaxScaler()
    X_tr_sc = scaler_kf.fit_transform(X_tr)
    X_te_sc = scaler_kf.transform(X_te)
    
    # Initialize the 4 models with optimal parameters from Task 2.4
    models_kf = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=None, min_samples_split=10, random_state=42),
        "SVM": SVR(C=100, gamma='scale'),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance')
    }
    
    # Train on 4 folds, test on the remaining 1 fold
    for name, model in models_kf.items():
        model.fit(X_tr_sc, Y_tr)
        preds = model.predict(X_te_sc)
        
        r2_kf[name].append(r2_score(Y_te, preds))
        mse_kf[name].append(mean_squared_error(Y_te, preds))
        mae_kf[name].append(mean_absolute_error(Y_te, preds))

# Convert fold results into DataFrames
fold_labels = [f"Fold {i}" for i in range(1, 6)]
df_r2_kf = pd.DataFrame(r2_kf, index=fold_labels)
df_mse_kf = pd.DataFrame(mse_kf, index=fold_labels)
df_mae_kf = pd.DataFrame(mae_kf, index=fold_labels)

print("\n--- R² Metric Across Each Fold ---")
display(df_r2_kf)
print("\n--- R² Summary Statistics Across 5 Folds (Mean, Std, Median, Min, Max) ---")
display(df_r2_kf.describe())

print("\n--- MSE Across Each Fold ---")
display(df_mse_kf)
print("\n--- MSE Summary Statistics Across 5 Folds ---")
display(df_mse_kf.describe())

print("\n--- MAE Across Each Fold ---")
display(df_mae_kf)
print("\n--- MAE Summary Statistics Across 5 Folds ---")
display(df_mae_kf.describe())



# %%

##Question 4 : Make conclusions about the best model and corresponding ML method (5-Fold CV)
best_kf_method_r2 = df_r2_kf.mean().idxmax()
best_kf_method_mse = df_mse_kf.mean().idxmin()
best_kf_method_mae = df_mae_kf.mean().idxmin()

best_single_fold_model = df_r2_kf.max().idxmax()
best_single_fold_name = df_r2_kf[best_single_fold_model].idxmax()
best_single_fold_score = df_r2_kf[best_single_fold_model].max()

print(f"Best ML Method on Average (R²):  {best_kf_method_r2} (Mean R² = {df_r2_kf.mean().max():.4f})")
print(f"Best ML Method on Average (MSE): {best_kf_method_mse} (Mean MSE = {df_mse_kf.mean().min():.4f})")
print(f"Best ML Method on Average (MAE): {best_kf_method_mae} (Mean MAE = {df_mae_kf.mean().min():.4f})")
print(f"Best Individual Fold Model:      {best_single_fold_model} on {best_single_fold_name} (R² = {best_single_fold_score:.4f})")

# Conclusions (Task 3.4):
# - Best ML Method Overall: Across all 5 disjoint folds, the non-linear methods (KNN / Decision Tree / SVM)
#   achieve the highest mean R² and lowest mean MSE/MAE, with the top method printed above ranking 1st on average.
# - Best Individual Model: Out of the 20 trained fold models (5 folds x 4 methods), the single highest R² score
#   is achieved by the model and fold printed above, while Linear Regression ranks last across every fold.


# %%

##Question 5 : Construct Bar Plots of Average Performance Results & Compare with Task 2.8
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
comparisons = [
    ("R² Score", df_r2_kf.mean(), df_r2_runs.mean()),
    ("MSE (Mean Squared Error)", df_mse_kf.mean(), df_mse_runs.mean()),
    ("MAE (Mean Absolute Error)", df_mae_kf.mean(), df_mae_runs.mean())
]

x = np.arange(len(model_names))
width = 0.36

for i, (metric_name, kf_means, rs_means) in enumerate(comparisons):
    rects1 = axes[i].bar(x - width/2, kf_means.values, width, label='5-Fold CV (Task 3)', color='#2b5c8f', edgecolor='black')
    rects2 = axes[i].bar(x + width/2, rs_means.values, width, label='5 Random Splits (Task 2.8)', color='#d95f02', edgecolor='black', alpha=0.85)
    
    axes[i].set_title(f"Average {metric_name}: 5-Fold CV vs. Task 2.8")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(model_names, rotation=15)
    axes[i].set_ylabel(f"Mean {metric_name}")
    axes[i].legend()
    axes[i].grid(True, axis='y', alpha=0.3)
    
    # Add numerical labels above bars
    for rect in rects1:
        h = rect.get_height()
        axes[i].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        axes[i].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Output side-by-side summary table comparing 5-Fold CV (Task 3) vs. 5 Random Splits (Task 2.8)
comp_task3_vs_28 = pd.DataFrame({
    "R² (5-Fold CV Mean)": df_r2_kf.mean(),
    "R² (5 Random Splits Mean)": df_r2_runs.mean(),
    "MSE (5-Fold CV Mean)": df_mse_kf.mean(),
    "MSE (5 Random Splits Mean)": df_mse_runs.mean(),
    "MAE (5-Fold CV Mean)": df_mae_kf.mean(),
    "MAE (5 Random Splits Mean)": df_mae_runs.mean()
})
print("\n=== Comparison of Average Performance: 5-Fold CV (Task 3) vs. 5 Random Splits (Task 2.8) ===")
display(comp_task3_vs_28)

# Conclusions (Task 3.5):
# 1. Agreement Between 5-Fold CV and 5 Random Splits:
#    - Both evaluation strategies produce nearly identical mean R², MSE, and MAE values and confirm the exact same
#      model tiering: Tuned KNN, Decision Tree, and SVM strongly outperform Linear Regression.
# 2. Advantage of 5-Fold Cross-Validation over Repeated Random Splits:
#    - In Task 2.8 (5 random splits), some observations are tested multiple times while others are never tested.
#    - In Task 3 (5-Fold CV), the dataset is partitioned into 5 strictly disjoint subsets so every country-year
#      observation is tested once and used for training 4 times, providing a more systematic, unbiased estimate
#      of generalization performance.


