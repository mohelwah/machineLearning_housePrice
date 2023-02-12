from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score,confusion_matrix

class neural_network:

    def __init__(self):
        pass
    def build_model(self,X_train, y_train):
        # initialize model
        print(X_train)
        self.model = Sequential()
        self.model.add(Dense(X_train.shape[1], activation='relu'))  # activation model : linear
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1))

        # build model

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.epochs_hist = self.model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.1)
        self.model.save('house_model.h5')
        #

    def build2_model(self):
        self.model = load_model('house_model.h5')

    def evaluate_model(self,X_test,y_test):
        # evaluate model
        y_pred = self.model.predict(X_test)

        expl_tr = explained_variance_score(y_pred, y_test)
        print("Explained variance score.....\n", expl_tr)
        print("Mean Absolute Error is ", mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error is", mean_squared_error(y_test, y_pred))
        print("Root Mean Squared Error is ", np.sqrt(mean_squared_error(y_test, y_pred)))
        print("Root Mean Squared Log Error", np.log(np.sqrt(mean_squared_error(y_test, y_pred))))
        print("R Squared", r2_score(y_test, y_pred))
        cutoff = 0.5
        y_pred_classes = np.zeros_like(y_pred)  # initialise a matrix full with zeros
        y_pred_classes[y_pred > cutoff] = 1  # add a 1 if the cutoff was breached

        y_test_classes = np.zeros_like(y_pred)
        y_test_classes[y_test > cutoff] = 1
        print("confusion matrix .....\n", confusion_matrix(y_test_classes, y_pred_classes))

    def predict(self,X_test1):
        # predict model
        #  Property','bathRoom','district','elevator','kitchen','livingRoom','size'
        #X_test1 = np.array([1, 5, 12, 1, 1, 3, 3500])

        y_pred_1 = self.model.predict(X_test1)[0]
        return y_pred_1

    def visual_(self):
        plt.plot(self.epochs_hist.history['loss'], color='b', label="Training loss")
        plt.show()
        plt.plot(self.epochs_hist.history['val_loss'], color='r', label="validation loss")
        plt.show()