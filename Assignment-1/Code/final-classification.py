# %% [markdown]
# # Task 1: Data Preprocessing

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# %%
#1) Read the data from your individual variant into the Python notebook.
data = pd.read_excel("10_Student_Performance.xlsx", sheet_name='student-por.csv')

# %%
#2) Output the main information (number of variables, number of instances).
data.info()
print("Shape:",data.shape)

# %%
#3) Output several first rows of the dataset.
data.head()

# %%
#4) Make explanatory data analyses for each variable (mean, std, etc.).
data.describe(include='all')


# %%
#5) Look at the independent and dependent variables. Divide the dataset accordingly into matrices X and Y.
# Output the shapes of these matrices.
# => We will convert the 'G3' from a numerical to binary 'pass/fail' (pass when G3 >= 10 and fail when G3 < 10)

data['Class'] = np.where(data.G3 >= 10, 'Pass', 'Fail')
Y = data.Class
X = data.drop(['Class', 'G3'], axis=1)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Number of unique classes:", Y.nunique())
Y.value_counts()

# %%
# 6) Look at data types of the variables. Are there any categorical variable you need to encode? Execute
# necessary transformations. Drop some columns if necessary, when recommended in your individual
# variant or the variables are not suitable for the prediction (e.g., Ids, person names).

# => we can easily encode the binary variables such as 'school', 'sex', 'adress' and so on, 
# and also the nominal variables, and we created 2 new variables from G1 and G2 that represent
# the average and the change between the two then we dropped them to avoid multicollinearity.

X['G_Avg'] = (X.G1 + X.G2) / 2
X['Grade_Change'] = X.G2 - X.G1
X = X.drop(['G1','G2'], axis=1)
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

X_encoded = pd.get_dummies(X,drop_first=True, dtype=int)
X_encoded.info()

# %%
# 7) Are there any Nan values? Try different filling of them (mean, median).
nan_count = X_encoded.isnull().sum().sum()
print("Total NaN valus in dataset :", nan_count)

X_filled_mean = X_encoded.fillna(X_encoded.mean(numeric_only=True))
X_filled_median = X_encoded.fillna(X_encoded.median(numeric_only=True))
print("Shapes after mean/median imputation:", X_filled_mean.shape, X_filled_median.shape)


# %%
# The dataset contains 0 missing (NaN) values, so mean and median imputation leave the data unchanged.

# %%
# 8) Check the dependences between each of the independent variable and the target variable and among
# the independent variables. Output correlation table and corresponding heatmap results.

corr_data = X_encoded.copy()
corr_data['Target_Class'] = Y_encoded

corr_table = corr_data.corr()
display(corr_table['Target_Class'].sort_values(ascending=False))
display(corr_table)
plt.figure(figsize=(20, 15))
sns.heatmap(corr_table, annot=False, cmap='coolwarm')
plt.show()

# %%
#9) Build 2D plots of dependences between each pair of the variables. Which conclusion can you make?
list_variables = corr_table['Target_Class'].abs().nlargest(6).index
sns.pairplot(corr_data[list_variables], hue='Target_Class')
plt.show()

# %%
# Conclusions : 
# - G_Avg is the strongest separator between classes, with failing students (class 0) concentrated 
# below 10 and passing students (class 1) above 10.
# - Students with higher studytime and plans for higher education (higher_yes = 1) achieve higher 
# G_Avg scores and predominantly pass, while students with multiple past failures cluster at lower G_Avg values and fail.

# %% [markdown]
# # Task 2: Data Analysis

# %%
# 1) Divide your dataset into 2 parts: training set and testing set (80:20). Use some specific random_state to
# be capable to repeat the used split of the data. Output the dimension of the resulting datasets.

X_train, X_test, Y_train, Y_test = train_test_split(
    X_encoded, Y_encoded, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# %%
# 2) Scale your datasets with MinMax Scaler.

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# 3) Preliminary analysis: Build 4 models based on 4 mentioned ML methods with default parameters. For
# each model: train with training set and output the regression score (R2), MSE, MAE of the model using
# training set. Now test your model with a testset and output the regression score (R2) and other metrics.
# Compare the metrics obtained with training and testset. Output dataframes with the following
# information. One dataframe for each metrics (e.g. 3 dataframes). Each row of the dataframe should
# correspond to a model and two columns correspond to the metric calculated on the training set and on
# the testset. Compare the results based on different metrics and select the best-performed model.
# Construct prediction squared errors histograms for each of the models, construct parity plots of each
# model.

models = {'Logistic Regression': LogisticRegression(), 
          'Decision Tree': DecisionTreeClassifier(random_state=42),
          'SVM': SVC(),
          'KNN': KNeighborsClassifier()}

accuracy_results = {}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, Y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)

    accuracy_results[name] = {
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc
    }

    cm = confusion_matrix(Y_test, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail (0)', 'Pass (1)'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f"{name}\n(Test Accuracy: {test_acc:.4f})")

plt.tight_layout()
plt.show()

# Construct prediction errors histograms for each of the models
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    test_pred = model.predict(X_test_scaled)
    errors = Y_test - test_pred  # 0 = Correct, -1 = False Positive, +1 = False Negative
    axes[i].hist(errors, bins=[-1.5, -0.5, 0.5, 1.5], rwidth=0.6, color='steelblue', edgecolor='black')
    axes[i].set_xticks([-1, 0, 1])
    axes[i].set_xticklabels(['-1 (False Pos)', '0 (Correct)', '+1 (False Neg)'])
    axes[i].set_title(f"{name} - Prediction Error Histogram")
    axes[i].set_ylabel("Number of Test Observations")

plt.tight_layout()
plt.show()


# Output DataFrame: rows = models, 2 columns = Train Accuracy and Test Accuracy
accuracy_df = pd.DataFrame(accuracy_results).T
display(accuracy_df)

#display(accuracy_df.plot(kind='bar', ylim=(0.75, 1.05), rot=0))



# %%
# Decision Tree has the best score, it reaches 100% training accuracy and 91.54% test accuracy, but 
# the most stable is the logistic regression model, from 89.79% on training accuracy to 90% on test accuracy.
# Meanwhile both SVM and KNN scored poorly compared to the other models.

# %%
# 4) Find optimal parameters of the above created models with GridSearchCV() function based on the
# training dataset. Output the optimal values of parameters found for each model. Output and construct
# bar plots of the coefficients of the linear regression model. Which variables have the biggest influence
# on the output (dependent variables)?

from sklearn.model_selection import GridSearchCV

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

param_grids = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
    'Decision Tree': {'max_depth': [2, 3, 4, 5, 7, 10], 'min_samples_split': [2, 5, 10]},
    'SVM': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
}

best_models = {}

for name, model in base_models.items():
    grid = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy')
    grid.fit(X_train_scaled, Y_train)
    
    best_models[name] = grid.best_estimator_
    print(f"{name} -> Optimal Parameters: {grid.best_params_} (CV Accuracy: {grid.best_score_:.4f})")

best_log_reg = best_models['Logistic Regression']
coef_df = pd.DataFrame({
    'Variable': X_encoded.columns,
    'Coefficient': best_log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

display(coef_df)

plt.figure(figsize=(12, 10))
sns.barplot(data=coef_df, x='Coefficient', y='Variable', palette='coolwarm')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title('Logistic Regression Coefficients (Feature Influence on Passing)')
plt.xlabel('Coefficient Value')
plt.ylabel('Independent Variables')
plt.tight_layout()
plt.show()


# %%
# - G_Avg (+35.36), Grade_Change (+3.49), and age (+3.35) have the strongest positive influence on passing.
# - Fjob_health (-4.12), Fjob_other (-2.30), Fjob_services (-2.07), and absences (-1.95) have the strongest negative influence.

# %%
# 5) Evaluate the performance of the models with optimal values of the parameters. Output the results in
# the similar dataframe as in Task 2.3. Compare the results with obtained in Task 2.3. Make the conclusion
# about the best-fitted model before and after optimization of the parameters.

optimized_results = {}

for name, model in best_models.items():
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    optimized_results[name] = {
        'Optimized Train': accuracy_score(Y_train, train_pred),
        'Optimized Test': accuracy_score(Y_test, test_pred)
    }

optimized_df = pd.DataFrame(optimized_results).T
comparison_df = pd.concat([accuracy_df, optimized_df], axis=1)
display(comparison_df)

#display(comparison_df[['Test Accuracy', 'Optimized Test']].plot(kind='bar', ylim=(0.75, 0.98), rot=0))


# %%
# Before hyperparameter tuning, the default Decision Tree achieved the highest test accuracy but 
# suffered from severe overfitting by memorizing the training data. After optimization with 
# cross-validation, Support Vector Machine with a linear kernel became the best-performing model 
# on the unseen test set, while pruning the Decision Tree and tuning Logistic Regression reduced 
# overfitting and produced reliable runners-up.

# %%
# 6) Output the prediction results of the models with optimal parameters: create a dataframe with 5
# columns, which correspond to "real values" and predicted values by each model on the testset and rows
# correspond to the observations from the testset. Construct a plot, which illustrates sample predictions
# for each model for randomly selected 10 observations from the testset (X-axis: observations, Y-axis:
# predicted value) and compare them to the real values. Output confusion matrix for each of the models.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

predictions_df = pd.DataFrame({'Real Values': Y_test}, index=X_test.index)
for name, model in best_models.items():
    predictions_df[name] = model.predict(X_test_scaled)

display(predictions_df.head(10))

sample_10 = predictions_df.sample(n=10, random_state=42)
display(sample_10)

plt.figure(figsize=(11, 5.5))
x_pos = np.arange(len(sample_10))
x_labels = [f"Obs {idx}" for idx in sample_10.index]

plt.scatter(x_pos, sample_10['Real Values'], color='black', marker='o', s=130, label='Real Values', zorder=5)
offsets = [-0.15, -0.05, 0.05, 0.15]
markers = ['^', 's', 'D', 'v']

for i, name in enumerate(best_models.keys()):
    plt.scatter(x_pos + offsets[i], sample_10[name], marker=markers[i], s=80, label=name, zorder=4)

plt.xticks(x_pos, x_labels)
plt.yticks([0, 1], ['0 (Fail)', '1 (Pass)'])
plt.xlabel('Randomly Selected Test Observations')
plt.ylabel('Class Value (Real vs. Predicted)')
plt.title('Sample Predictions vs. Real Values for 10 Random Test Observations')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, model) in enumerate(best_models.items()):
    cm = confusion_matrix(Y_test, predictions_df[name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail (0)', 'Pass (1)'])
    disp.plot(ax=axes[i], cmap='Greens', colorbar=False)
    axes[i].set_title(f"Optimized {name}")

plt.tight_layout()
plt.show()


# %%
# 7) Calculate the prediction errors of each observation from the testset and obtained by each model. Make
# conclusions about the predictions, which have the biggest prediction error. Compare the prediction
# errors. Are there any observations badly predicted by all the methods?

errors_df = pd.DataFrame(index=X_test.index)
for name in best_models.keys():
    errors_df[name] = (predictions_df[name] != predictions_df['Real Values']).astype(int)

errors_df['Total_Models_Wrong'] = errors_df.sum(axis=1)

print("Misclassified observations per model:")
print(errors_df.sum())

print("\nBreakdown of observations by number of models wrong:")
print(errors_df['Total_Models_Wrong'].value_counts().sort_index())

# Filter observations wrongly classified by all 4 models
all_wrong_idx = errors_df[errors_df['Total_Models_Wrong'] == 4].index
all_wrong_details = pd.concat([
    predictions_df.loc[all_wrong_idx],
    data.loc[all_wrong_idx, ['G1', 'G2', 'G3', 'failures', 'absences']]
], axis=1)

print("\nObservations wrongly classified by all models:")
display(all_wrong_details)


# %%
# SVM produced the fewest classification errors on the test set, Logistic Regression 
# and Decision Tree tied for second, and KNN made the most mistakes because it predicted 
# every student would pass. Only two borderline students were misclassified by all four 
# models, as both had prior trimester grades near the passing cutoff before experiencing 
# an unexpected drop in their final exam score.

# %%
# 8) Change the random_state in task 2.1 and re-calculate all the previous steps. Are your conclusions about
# the best model the same? Make conclusions averaging the performance results for to five different
# random states. Construct the BoxPlots for each ML method and for each of the performance metrics,
# corresponding to the five different runs. Construct the Bar Plots for each ML method and for each of
# the performance metrics with average performance results. Describe and interpret the results.

# 8) Re-calculate previous steps across 5 different random_state values
random_states = [5, 15, 25, 35, 42]
run_records = []

for rs in random_states:
    # Split and scale data for the current random_state
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_encoded, Y_encoded, test_size=0.2, random_state=rs
    )
    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=rs),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    }
    
    # Tune hyperparameters and record test accuracy for each model
    for name, model in base_models.items():
        grid = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy')
        grid.fit(X_tr_sc, Y_tr)
        best_m = grid.best_estimator_
        
        run_records.append({
            'Random_State': rs,
            'Model': name,
            'Train_Accuracy': accuracy_score(Y_tr, best_m.predict(X_tr_sc)),
            'Test_Accuracy': accuracy_score(Y_te, best_m.predict(X_te_sc))
        })

runs_df = pd.DataFrame(run_records)
pivot_test = runs_df.pivot(index='Random_State', columns='Model', values='Test_Accuracy')
display(pivot_test)

# Display average performance summary
summary_stats = pivot_test.agg(['count', 'mean', 'median', 'min', 'max', 'std']).T
display(summary_stats)

# Construct BoxPlots and Bar Plots for the 5 runs
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
model_order = ['Logistic Regression', 'Decision Tree', 'SVM', 'KNN']

sns.boxplot(data=runs_df, x='Model', y='Test_Accuracy', order=model_order, ax=axes[0])
axes[0].set_title('BoxPlot of Test Accuracy Across 5 Random States')
axes[0].set_ylabel('Test Accuracy')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

sns.barplot(data=runs_df, x='Model', y='Test_Accuracy', order=model_order, errorbar='sd', capsize=0.1, ax=axes[1])
axes[1].set_title('Bar Plot of Average Test Accuracy (5 Runs)')
axes[1].set_ylabel('Average Test Accuracy')
axes[1].set_ylim(0.75, 0.98)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

runs_melted = runs_df.melt(
    id_vars=['Random_State', 'Model'],
    value_vars=['Train_Accuracy', 'Test_Accuracy'],
    var_name='Metric',
    value_name='Accuracy'
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
model_order = ['Logistic Regression', 'Decision Tree', 'SVM', 'KNN']

sns.boxplot(data=runs_melted, x='Model', y='Accuracy', hue='Metric', order=model_order, ax=axes[0])
axes[0].set_title('BoxPlot of Train & Test Accuracy Across 5 Random States')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

sns.barplot(data=runs_melted, x='Model', y='Accuracy', hue='Metric', order=model_order, errorbar='sd', capsize=0.08, ax=axes[1])
axes[1].set_title('Bar Plot of Average Train & Test Accuracy (5 Runs)')
axes[1].set_ylim(0.75, 1.03)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



# %%
# Changing the random seed shifts the winning model: although SVM ranked first on the single 
# split in the earlier step, averaging across multiple random states reveals that the tuned 
# Decision Tree achieves the highest mean test accuracy and the lowest variance. Logistic Regression 
# ranks a close second and reaches the highest individual run score, SVM places third due to a dip on 
# one split, and KNN consistently trails with the lowest average accuracy and widest spread.

# %% [markdown]
# # Task 3: Kfold cross validation

# %%
# 1) Use the functions Kfold to split your data into 5 disjoint datasets. Output the indexes of the observations
# in each splited dataset.

from sklearn.model_selection import KFold

# Initialize 5-fold cross-validator with shuffling for balanced splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop through the 5 splits and output the observation indexes for each disjoint dataset
for fold, (train_idx, test_idx) in enumerate(kf.split(X_encoded), start=1):
    print(f"=== Disjoint Dataset {fold} (Size: {len(test_idx)}) ===")
    print("Observation Indexes:", test_idx)
    print()


# %%
# 2) Build 5 models for each of the ML methods, each time taking one dataset as a testset and joint remaining
# datasets as a training set. Use the best hyper-parameter values for each model found in task 2.

# Prepare containers to store the 5 trained models per method and their accuracy scores
trained_kfold_models = {
    'Logistic Regression': [],
    'Decision Tree': [],
    'SVM': [],
    'KNN': []
}
fold_records = []

# Loop through the 5 disjoint splits generated by KFold
for fold, (train_idx, test_idx) in enumerate(kf.split(X_encoded), start=1):
    # 1. Extract training (4 subsets joined) and testing (1 subset) data for this fold
    X_tr_fold = X_encoded.iloc[train_idx]
    X_te_fold = X_encoded.iloc[test_idx]
    Y_tr_fold = Y_encoded[train_idx]
    Y_te_fold = Y_encoded[test_idx]
    
    # 2. Scale features strictly inside each fold to prevent data leakage
    scaler = MinMaxScaler()
    X_tr_fold_sc = scaler.fit_transform(X_tr_fold)
    X_te_fold_sc = scaler.transform(X_te_fold)
    
    # 3. Initialize the 4 models using the optimal hyperparameters from Task 2.4
    opt_models_fold = {
        'Logistic Regression': LogisticRegression(C=100, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, min_samples_split=2, random_state=42),
        'SVM': SVC(C=100, kernel='linear'),
        'KNN': KNeighborsClassifier(n_neighbors=11, weights='distance')
    }
    
    # 4. Fit each model on the current fold's training set and save the results
    for name, model in opt_models_fold.items():
        model.fit(X_tr_fold_sc, Y_tr_fold)
        trained_kfold_models[name].append(model)
        
        tr_acc = accuracy_score(Y_tr_fold, model.predict(X_tr_fold_sc))
        te_acc = accuracy_score(Y_te_fold, model.predict(X_te_fold_sc))
        
        fold_records.append({
            'Fold': f"Fold {fold}",
            'Model': name,
            'Train_Accuracy': tr_acc,
            'Test_Accuracy': te_acc
        })

# Verify that 5 models were built for each ML method
for name, model_list in trained_kfold_models.items():
    print(f"{name}: {len(model_list)} models trained")

# Convert records into a DataFrame and preview Training Accuracy across folds
fold_df = pd.DataFrame(fold_records)
pivot_kfold_train = fold_df.pivot(index='Fold', columns='Model', values='Train_Accuracy')
display(pivot_kfold_train)


# %%
# 3) Evaluate the models’ performance of each fold for each ML method separately on the base of accuracy metric.
# Output the results of each fold and compute an average value of the metric.

# 1. Output the Test Accuracy results of each fold for each ML method
pivot_kfold_test = fold_df.pivot(index='Fold', columns='Model', values='Test_Accuracy')
print("Test Accuracy for each fold:")
display(pivot_kfold_test)

# 2. Compute and output the average value (and summary statistics) of the metric across the 5 folds
kfold_summary = pivot_kfold_test.agg(['count', 'mean', 'median', 'min', 'max', 'std']).T
print("\nAverage Test Accuracy across the 5 folds:")
display(kfold_summary[['mean', 'std', 'min', 'max']])

# 3. Output the Confusion Matrix for each of the folds (5 folds x 4 ML methods)
model_order = ['Logistic Regression', 'Decision Tree', 'SVM', 'KNN']
fig, axes = plt.subplots(5, 4, figsize=(16, 18))

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_encoded)):
    scaler = MinMaxScaler()
    scaler.fit(X_encoded.iloc[train_idx])
    X_te_fold_sc = scaler.transform(X_encoded.iloc[test_idx])
    Y_te_fold = Y_encoded[test_idx]
    
    for col_idx, name in enumerate(model_order):
        model = trained_kfold_models[name][fold_idx]
        preds = model.predict(X_te_fold_sc)
        cm = confusion_matrix(Y_te_fold, preds)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail (0)', 'Pass (1)'])
        disp.plot(ax=axes[fold_idx, col_idx], cmap='Blues', colorbar=False)
        axes[fold_idx, col_idx].set_title(f"Fold {fold_idx + 1}: {name}")

plt.tight_layout()
plt.show()


# %%
# 4) Make conclusions about the best model and corresponding ML method.

# Identify the best ML method (highest average accuracy) and best individual fold model
best_method = kfold_summary['mean'].idxmax()
best_method_score = kfold_summary['mean'].max()

best_single_row = fold_df.loc[fold_df['Test_Accuracy'].idxmax()]
print(f"Best ML Method on average: {best_method} (Mean Test Accuracy: {best_method_score:.4f})")
print(f"Best individual model: {best_single_row['Model']} on {best_single_row['Fold']} (Test Accuracy: {best_single_row['Test_Accuracy']:.4f})")

# Conclusions:
# - Best ML Method: Logistic Regression is the best-performing ML method overall across the 5 folds, 
#   achieving the highest average test accuracy of 91.99% (0.9199), followed closely by the tuned 
#   Decision Tree at 91.22% (0.9122) and SVM at 91.06% (0.9106), while KNN performs the worst at 84.44% (0.8444).
# - Best Individual Model: Out of the 20 trained fold models, the Logistic Regression model trained on Fold 2 
#   achieves the highest single-fold test accuracy of 94.62% (0.9462), while the pruned Decision Tree shows 
#   the highest stability across folds with the lowest standard deviation (0.0041).


# %%
# 5) Construct the Bar Plots for each ML method and for each of the performance metrics with average
# performance results and compare the results with Task 2.8. Make conclusions.

# 1. Construct the Bar Plot of average Test Accuracy across the 5 folds for each ML method
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.barplot(data=fold_df, x='Model', y='Test_Accuracy', order=model_order, errorbar='sd', capsize=0.1, ax=axes[0])
axes[0].set_title('Task 3.5: Average K-Fold Test Accuracy (5 Folds)')
axes[0].set_ylabel('Average Test Accuracy')
axes[0].set_ylim(0.75, 0.98)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# 2. Compare Task 3 (5-Fold CV) average results with Task 2.8 (5 Random Splits)
comparison_2_8_vs_3 = pd.DataFrame({
    'Task 2.8 Mean (5 Random Splits)': summary_stats['mean'],
    'Task 2.8 Std': summary_stats['std'],
    'Task 3 Mean (5-Fold CV)': kfold_summary['mean'],
    'Task 3 Std': kfold_summary['std']
}).loc[model_order]

display(comparison_2_8_vs_3)

# Plot side-by-side comparison of Task 2.8 vs. Task 3
plot_comp_df = comparison_2_8_vs_3[['Task 2.8 Mean (5 Random Splits)', 'Task 3 Mean (5-Fold CV)']].reset_index()
plot_comp_df = plot_comp_df.melt(id_vars='Model', var_name='Evaluation Method', value_name='Average Test Accuracy')

sns.barplot(data=plot_comp_df, x='Model', y='Average Test Accuracy', hue='Evaluation Method', order=model_order, ax=axes[1])
axes[1].set_title('Comparison of Average Test Accuracy: Task 2.8 vs. Task 3')
axes[1].set_ylim(0.75, 0.98)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.show()

# Conclusions:
# - In Task 2.8 (5 random splits), Decision Tree ranked first (92.62%) and Logistic Regression ranked second (91.85%), 
#   whereas in Task 3 (5-fold cross-validation), Logistic Regression ranks first (91.99%) and Decision Tree ranks second (91.22%).
# - Both evaluation methods confirm that Logistic Regression, Decision Tree, and SVM (~90.3%–92.6%) strongly outperform KNN (~84.4%–85.1%).
# - 5-Fold Cross-Validation (Task 3) produces much lower standard deviations across folds for Decision Tree (0.0041 vs. 0.0117), 
#   SVM (0.0088 vs. 0.0235), and Logistic Regression (0.0159 vs. 0.0215) than the 5 random splits in Task 2.8, because 
#   disjoint folds guarantee that every student observation is tested exactly once without overlapping test sets.



