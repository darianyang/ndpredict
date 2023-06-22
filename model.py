"""
Building a RF model for N->D deamidation probability prediction.

TODO: 
    * feature selection using RFE: recursive feature elimination.
        * they didn't implement this into their model, they just did it for the paper.
    * LDA could be useful here? Can predict labels or label probabilities for ROCAUC.
    * check for overfitting: ROCAUC curve for training vs test data
    * plot CV of data: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html


RF ROCAUC from paper: 0.96
Using the out-of-the-box sklearn RF regressor on same feature set: ROCAUC is 0.87-0.89
I made the following changes to feature space:
    * secondary_structure is 1/2/3/4, these values are assumed to be ordinal in
        the original paper/model, but they would be better maybe as one-hot encoded
    * The dihedral angles are in degrees, since these are periodic values this could
        be improved by converting to sin and cos dimensions.
    * this did not change the ROCAUC, likely because it is very dependent on Half_life
I did hyperparameter opt of RF with grid search and CV:
    * no significant improvements from the default parameters
I followed up with manual hyperparameter tuning to reduce overfitting based on ROCAUC of
training vs test datasets, these new parameters seem to be better based on CV scores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import mdtraj

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class NDPredict:
    """
    Building a ML model for N->D deamidation probability prediction.
    """

    def __init__(self, csv="data/fulldata.csv", train_csv="data/train.csv", test_csv="data/test.csv"):
        """
        Parameters
        ----------
        csv : str
            Path to input csv data file with entire dataset.
        train_csv : str
            Just the training data.
        test_csv : str
            Just the test data.
        """
        self.csv = csv
        self.train_csv = train_csv
        self.test_csv = test_csv

    @staticmethod
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

    def proc_csv(self, csv):
        """
        Take the input dataset and output a pandas df.
        Process secondary structure and dihedral angle features. 
        Standardize feature dataset, encode label dataset.
        
        Parameters
        ----------
        csv : str
            Path to input csv data file.

        Returns
        -------
        X : 2darray
            X is a 2d array of standardized and processed features.
        y : 2darray
            y is a 2d array of encoded labels.
        self.feat_names : 1darray
            Array of feature names.
        """
        # Load the dataset from CSV
        data = pd.read_csv(csv)

        # Separate features and target variable
        # All features except the last one (labels), and skip PDB/res#/AAfollowing cols
        # these cols were not used in the paper
        X = data.iloc[:, 3:-1]
        # Target variable (last column = deamidation : yes or no)
        y = data.iloc[:, -1] 

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
            converted_data = self.proc_angle_data(X[column])
            sincos_lookup = {0:"cos", 1:"sin"}
            for i in range(converted_data.shape[1]):
                new_column_name = f"{column}_{sincos_lookup[i]}"
                X[new_column_name] = converted_data[:, i]

        # drop original angle features
        X = X.drop(columns_to_convert, axis=1)

        # save feature names
        self.feat_names = X.columns.values

        # Standardize the features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)

        # Encode the target variable from yes/no to binary 1/0
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(y)

        return self.X, self.y, self.feat_names

    def data_split(self, use_train_test=True):
        """
        Take self.X and self.y, return updates instance attributes
        self.X_train, self.X_test, self.y_train, self.y_test.

        Parameters
        ----------
        use_train_test : bool
            Be default False, uses `csv`, True will use `train_csv` and `test_csv`.
        """
        # use the specifically definied training and test csv files
        # TODO: could replace this and args by just taking the bottom n values of full csv
        if use_train_test:
            self.X_train, self.y_train, _ = self.proc_csv(self.train_csv)
            self.X_test, self.y_test, _ = self.proc_csv(self.test_csv)
            # Split the full dataset into training and testing sets
            self.proc_csv(self.csv)
        else:
            # Split the full dataset into training and testing sets
            self.proc_csv(self.csv)
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(self.X, self.y, test_size=0.3, random_state=1)

    def rocauc_score(self, model, plot=True):
        """
        Input model (e.g. RF regressor) and output ROCAUC score.
        Saves the model as self.model.

        Parameters
        ----------
        model : sklearn model object
        plot : bool
            By default (True), outputs a ROC curve plot.
        """
        # make plot object
        fig, ax = plt.subplots()

        # fit the model
        model.fit(self.X_train, self.y_train)

        # plot rocauc of training data
        y_pred = model.predict(self.X_train)
        roc_auc = roc_auc_score(self.y_train, y_pred)
        print("TRAIN ROC AUC score:", roc_auc)
        if plot:
            fpr, tpr, _ = roc_curve(self.y_train, y_pred)
            self.plot_roc_curve(fpr, tpr, roc_auc, "Train", ax=ax)

        # Predict the probabilities on the test set
        y_pred = model.predict(self.X_test)

        # Compute ROC AUC score
        roc_auc = roc_auc_score(self.y_test, y_pred)

        print("TEST ROC AUC score:", roc_auc)
        #print(dict(zip(feat_names, model.feature_importances_)))

        # ROC AUC plot (TODO: seperate methods for plots)
        if plot:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred)
            self.plot_roc_curve(fpr, tpr, roc_auc, "Test", ax=ax)

        # save the model and print oob score if avail
        self.model = model
        if hasattr(self.model, "oob_score_"):
            print("Score: ", self.model.score(self.X_test, y_pred))
            print("OOB Score: ", self.model.oob_score_)

    def cv_score(self, n_fold=10):
        """
        n fold CV score of the model.
        """
        # scores = cross_val_score(self.model, self.X, self.y, scoring="roc_auc", cv=n_fold)
        # print("CV: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        scores = cross_validate(self.model, self.X, self.y, 
                                scoring="roc_auc", cv=n_fold, return_train_score=True)
        print(f"CV TEST: {np.mean(scores['test_score'])}, CV TRAIN: {np.mean(scores['train_score'])}")

    def plot_feat_importances(self):
        """
        Simple bar plot of feature importances from RF model.
        """
        plt.bar(self.feat_names, self.model.feature_importances_)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def rf_grid_opt(self):
        """
        Optimize RF model hyperparameters.
        Results point to defaults being the best.
        """
        # create grid
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
        # The function to measure the quality of a split
        #criterion = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
        # Number of features to consider at every split
        max_features = [1.0, 'sqrt', 'log2', 0.33]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the grid
        grid = {'n_estimators': n_estimators,
                #'criterion' : criterion,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Search of parameters, using 3 fold cross validation, 
        # search across 80 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=80, cv=3, 
                                    verbose=1, n_jobs=8, scoring="roc_auc")
        #rf_grid = GridSearchCV(estimator=rf, param_grid=grid, cv=3, verbose=1, n_jobs=8, scoring="roc_auc")

        # Fit the random search model
        rf_random.fit(self.X_train, self.y_train)

        # print and test out best parameters
        print(rf_random.best_params_)
        #best = {'n_estimators': 110, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None, 'bootstrap': True}
        self.rocauc_score(RandomForestRegressor(**rf_random.best_params_))

    def plot_roc_curve(self, x, y, score, label="", ax=None):
        """
        Function for plotting the reciever operator characteristic curve 
        with the X axis as the false positive rate and the y axis as the 
        true positive rate.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        ax.plot(x, y, label=f"{label}AUC: {score:0.3f}")
        ax.plot([0, 1], [0, 1], color="k", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend()

    def get_rf_model(self):
        """
        Train and return a RF model using the input data.
        """
        pass

    def pdb_proc(self, pdb, asns=None):
        """
        Input a pdb file and calculate ML features using MDTraj.

        Parameters
        ----------
        pdb : str
            Path to PDB file input. Make sure this file is protonated 
            fully cleaned up.
        asns : list of ints
            List of residue numbers to select which ASN residues to calculate
            features for. If deafult None, all ASN residues found will be used.

        Returns
        -------
        pdb_feats : 2darray
            Array with each row as an ASN and each column as features.
        """
        # load in cleaned PDB file

        # make df or array for features

        # columns: PDB	Residue #	AA following Asn	attack_distance	Half_life	norm_B_factor_C	norm_B_factor_CA	norm_B_factor_CB	norm_B_factor_CG	secondary_structure	PSA	PSSA	Psi	Phi	Chi1	Chi2	Deamidation

        # use self.feat_names for column names if needed for df

        pass

    def pdb_pred(self):
        """
        Use the input PDB feature set to calc N->D probabilities?
        Or combine into a single method.
        """
        pass

if __name__ == "__main__":
    # ndp = NDPredict()
    # ndp.data_split(use_train_test=True)
    # #ndp.rf_grid_opt()
    # # opt to prevent overfitting training data
    # best = {'n_estimators': 1000, 'min_samples_split': 100, 'min_samples_leaf': 10, 'max_features': 0.33, 'max_depth': 3, 'bootstrap': True, 'oob_score' : True}
    # ndp.rocauc_score(RandomForestRegressor(**best), plot=False)
    # #ndp.rocauc_score(RandomForestRegressor(oob_score=True), plot=False)
    # ndp.cv_score(5)    
    #plt.show()
    # # save the ndp object to a pickel object
    # pickle.dump(ndp, open("ndp_model.pkl", 'wb'))

    # load pickel model object
    ndp = pickle.load(open("ndp_model.pkl", 'rb'))
    ndp.plot_feat_importances()
    plt.show()
    
