"""
Building a RF model for N deamidation probability prediction.

TODO: 
    * feature selection using RFE: recursive feature elimination.
    * LDA could be useful here? Can predict labels or label probabilities for ROCAUC.
    * hyperparameter opt of RF and/or LDA
    * do 10 fold CV to account for bad train/test splitting

RF ROCAUC from paper: 0.96
Using the out-of-the-box sklearn RF regressor on same feature set: ROCAUC is 0.87-0.89
I made the following changes to feature space:
    * secondary_structure is 1/2/3/4, these values are assumed to be ordinal in
        the original paper/model, but they would be better maybe as one-hot encoded
    * The dihedral angles are in degrees, since these are periodic values this could
        be improved by converting to sin and cos dimensions.
    * this did not improve the ROCAUC, likely because it is very dependent on Half_life
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset from CSV
data = pd.read_csv('data/data.csv')

# Separate features and target variable
# All features except the last one (labels), and skip PDB/res#/AAfollowing cols
# these cols were not used in the paper
X = data.iloc[:, 3:-1]
# Target variable (last column = deamidation : yes or no)
y = data.iloc[:, -1] 

def proc_angle_data(data):
    """
    Periodic angles (e.g. dihedrals) near periodic boundaries will
    behave poorly since -179° and 179° are actually only 2° away.
    This converts a single angle to radians, then returns the 
    sin and cos of the radians.
    """
    # conver to rads
    data_rad = data * np.pi / 180
    # convert to cos and sin
    data_rad_cos = np.cos(data_rad)
    data_rad_sin = np.sin(data_rad)
    # stack sin and cos arrays:
    data_rad_cos_sin = np.column_stack((data_rad_cos, data_rad_sin))
    return data_rad_cos_sin

### Feature processing ###
# one hot encoding of secondary_structure column
# Extract the amino acid residue feature
residue_feature = X['secondary_structure']
# Initialize the one-hot encoder (return array, not sparse matrix)
encoder = OneHotEncoder(sparse=False)
# Fit and transform the residue feature
residue_encoded = encoder.fit_transform(residue_feature.values.reshape(-1, 1))
# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(residue_encoded, columns=encoder.get_feature_names_out(['secondary_structure']))
# Concatenate the encoded features with the original dataset
X = pd.concat([X.drop('secondary_structure', axis=1), encoded_df], axis=1)

# convert periodic dihedral angle values
# Convert "phi," "psi," "chi1," and "chi2" columns
columns_to_convert = ["Phi", "Psi", "Chi1", "Chi2"]

# make sin/cos processed dihedral data and add to df
for column in columns_to_convert:
    converted_data = proc_angle_data(X[column])
    sincos_lookup = {0:"cos", 1:"sin"}
    for i in range(converted_data.shape[1]):
        new_column_name = f"{column}_{sincos_lookup[i]}"
        X[new_column_name] = converted_data[:, i]

# drop original angle features
X = X.drop(columns_to_convert, axis=1)

# save feature names
feat_names = X.columns.values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the target variable from yes/no to binary 1/0
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = \
    train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=1)
def rocauc_score(model):
    """
    Input model (e.g. RF regressor) and output ROCAUC score.
    """
    # fit the model
    model.fit(X_train, y_train)

    # Predict the probabilities on the test set
    y_pred = model.predict(X_test)

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred)

    print("ROC AUC score:", roc_auc)
    #print(dict(zip(feat_names, model.feature_importances_)))

    plt.bar(feat_names, model.feature_importances_)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

#rocauc_score(RandomForestRegressor(random_state=1))
rocauc_score(RandomForestClassifier(random_state=1))

# # RF parameter opt
# from sklearn.model_selection import RandomizedSearchCV
# # create random grid
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=80, cv=3, verbose=1, n_jobs=8)

# # Fit the random search model
# rf_random.fit(X_train, y_train)

# # save and test out best parameters
# print(rf_random.best_params_)
best = {'n_estimators': 1600, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}
#best = {'n_estimators': 1800, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': False}
#rocauc_score(RandomForestRegressor(**rf_random.best_params_))
#rocauc_score(RandomForestRegressor(**best))
rocauc_score(RandomForestClassifier(**best))
