import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import explained_variance_score, confusion_matrix,mean_absolute_error,mean_squared_error,r2_score,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from joblib import dump,load
class machine_learning:
    def __init__(self,random_state=0,splitter="best"):
        self.random_state = random_state
        self.splitter = splitter

    # initialize model
    def Build_m(self, X_train, y_train):
            self.tr_regressor = DecisionTreeRegressor(splitter=self.splitter, random_state=self.random_state)
            self.tr_regressor.fit(X_train, y_train)
            dump(self.tr_regressor, 'machinelearning.joblib')

    def Build2_m(self):
            self.tr_regressor = load('machinelearning.joblib')

    def predict(self, X_test):
            pred_tr = self.tr_regressor.predict(X_test)[0]
            return pred_tr

    def eval_m(self, X_test, y_test):
        print("Decision tree  Regression Model Score is ", round(self.tr_regressor.score(X_test, y_test) * 100))
        pred_tr = self.tr_regressor.predict(X_test)
        expl_tr = explained_variance_score(pred_tr, y_test)
        print("Explained variance regression score.....\n", expl_tr)
        print("Mean Absolute Error is ", mean_absolute_error(y_test, pred_tr))
        print("Mean Squared Error is", mean_squared_error(y_test, pred_tr))
        print("Root Mean Squared Error is ", np.sqrt(mean_squared_error(y_test, pred_tr)))
        print("Root Mean Squared Log Error", np.log(np.sqrt(mean_squared_error(y_test, pred_tr))))
        print("R Squared", r2_score(y_test, pred_tr))

        cutoff = 0.5  # decide on a cutoff limit
        y_pred_classes = np.zeros_like(pred_tr)  # initialise a matrix full with zeros
        y_pred_classes[pred_tr > cutoff] = 1  # add a 1 if the cutoff was breached

        y_test_classes = np.zeros_like(pred_tr)
        y_test_classes[y_test > cutoff] = 1

        print("confusion matrix .....\n", confusion_matrix(y_test_classes, y_pred_classes))
        print("Classification report...\n", classification_report(y_test_classes, y_pred_classes))

    def visual_(self):
        plt.figure(figsize=(12, 8))
        tree.plot_tree(self.tr_regressor,
                       feature_names=['Property', 'bathRoom', 'district', 'elevator', 'kitchen', 'livingRoom', 'size'],
                       filled=True, fontsize=10)

        plt.show()