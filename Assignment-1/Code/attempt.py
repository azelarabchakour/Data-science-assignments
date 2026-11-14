# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# %%
#1) Read the data from your individual variant into the Python notebook.
data = pd.read_excel("10_Student_Performance.xlsx")

# %%
#2) Output the main information (number of variables, number of instances).
data.info()
data.shape

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

# => let's drop the 'G1' and 'G2' variables because it makes no sense if we keep them, the model will have 
# always a 100% accuracy, then for the encoding, we can easily encode the binary variables such as 'school',
# 'sex', 'adress' and so on, and also the nominal variables

X = X.drop(['G1','G2'], axis=1)
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

X_encoded = pd.get_dummies(X,drop_first=True, dtype=int)
X_encoded.info()

# %%
# 7) Are there any Nan values? Try different filling of them (mean, median).
X_encoded.isnull().sum().sum()

# %%
# 8) Check the dependences between each of the independent variable and the target variable and among
# the independent variables. Output correlation table and corresponding heatmap results.

corr_data = X_encoded.copy()
corr_data['Target_Class'] = Y_encoded

corr_table = corr_data.corr()
display(corr_table)
plt.figure(figsize=(20, 15))
sns.heatmap(corr_table, annot=False, cmap='coolwarm')
plt.show()

# %%



