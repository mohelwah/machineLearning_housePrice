import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
class Dataset_:
    def __init__(self,path_china,path_ksa):
        self.china = path_china
        self.KSA   = path_ksa

    def read_(self):
        self.china = pd.read_csv(self.china)
        self.KSA   = pd.read_csv(self.KSA)

    def drop_unwanted_cols(self):
        china_cols = ['Unnamed: 0', 'tradeTime', 'totalPrice', 'drawingRoom','floor', 'buildingType', 'constructionTime', 'renovationCondition', 'buildingStructure','ladderRatio', 'subway', 'communityAverage']
        ksa_cols   = ['city', 'front', 'bedrooms','garage', 'basement', 'driver_room', 'maid_room', 'furnished', 'ac', 'roof', 'pool', 'frontyard','duplex', 'stairs', 'fireplace', 'details']

        self.china = self.china .drop(china_cols, axis=1)
        self.KSA   = self.KSA.drop(ksa_cols, axis=1)

    def handling_missing_values(self):
       print("missing values in china")
       print(self.china.isna().sum()) # sum to calculate missing value

       print("missing values in KSA")
       print(self.KSA.isna().sum())

        # replace missing value with mean
       self.china = self.china.fillna(self.china.mean())
       self.KSA = self.KSA.fillna(self.KSA.mean())

    def delete_space(self,text): # to delete spaces from word in KSA dataset which writtin in Arabic so we could convert it to int
        return text.strip()

    def convert_string_to_int(self,Col_data, distinct_arr, number_start):#these method convert some columns in the datasets from string to int (word to numbers)
        # This function receives the name of the column containing the words, an array of the
        # words found without repetition, and the number with which the loop will start
        for v in distinct_arr:
            Col_data = Col_data.replace([v], number_start)
            number_start += 1
        return Col_data

    def district(self):
        self.KSA['district'] = self.KSA['district'].apply(self.delete_space)
        self.district_china  = self.china['district'].unique()
        self.district_KSA    = self.KSA['district'].unique()

    def encoding_(self):# these method well apply method delete_space then convert_string_to_int
        self.KSA['district'] = self.KSA['district'].apply(self.delete_space)

        self.district_china = self.china['district'].unique()
        self.district_KSA   = self.KSA['district'].unique()

        count = self.KSA['district'].nunique() #unique() function makes a list of existing words without repetition then we could convert it to int
        # convert to int using convert_string_to_int method
        self.KSA['district']   = self.convert_string_to_int(self.KSA['district'], self.district_KSA, 0)
        self.china['district'] = self.convert_string_to_int(self.china['district'], self.district_china, count+1)
        # where() function change the condition to 0 otherwise 1
        self.china["elevator"] = np.where(self.china["elevator"] == "No_elevator", 0, 1)
        self.KSA["elevator"]   = np.where(self.KSA["elevator"] == "No_elevator", 0, 1)
        # fiveYearsProperty column in china dataset the values is 0 if it's more tan 5 years otherwise 1, in KSA datasets
        # represent with numbers, So we set a condition for the column if the values are less than 5 put 0 otherwise 1
        self.china["fiveYearsProperty"] = np.where(self.china["fiveYearsProperty"] == "Ownership < 5y", 0, 1)
        self.KSA["property_age"]        = np.where(self.KSA["property_age"] < 5, 0, 1)

    def handling_outlier(self):
        numeric_col  = ['size', 'livingrooms', 'price']
        numeric_col2 = ['square', 'livingRoom', 'price']
        #plt.boxplot(self.KSA['bathrooms'])
        #plt.show()
        # use IQR SCORE TO filter out the outliers by keeping only valid values.
        for x,y in zip(numeric_col,numeric_col2):
            q1, q3 = np.percentile(self.KSA.loc[:, x], [75, 25])
            q11, q13 = np.percentile(self.china.loc[:, y], [75, 25])

            IQR  = q1 - q3
            IQR1 = q11 - q13

            max = q1 + (1.5 * IQR)
            min = q3 - (1.5 * IQR)

            max1 = q11 + (1.5 * IQR1)
            min1 = q13 - (1.5 * IQR1)

            self.KSA.loc[self.KSA[x] < min, x] = self.KSA[x].mean()
            self.KSA.loc[self.KSA[x] > max, x] = self.KSA[x].mean()

            self.china.loc[self.china[y] < min, y] = self.china[y].mean()
            self.china.loc[self.china[y] > max, y] = self.china[y].mean()

    def Data_normalize(self):
        # normalize x-min/max-min
        self.china = (self.china - self.china.mean()) / self.china.std()
        self.KSA = (self.KSA - self.KSA.mean()) / self.KSA.std()

    def merage_Dataset(self):
        d1 = self.china.rename(columns={"square": "size", "fiveYearsProperty": "Property"})
        d2 = self.KSA.rename(columns={"property_age": "Property", "bathrooms": "bathRoom", "livingrooms": "livingRoom"})
        cols = ['Property', 'bathRoom', 'district', 'elevator', 'kitchen', 'livingRoom', 'size', 'price']

        self.data = pd.concat([d1[cols], d2[cols]])
        self.data['bathRoom'] = self.data['bathRoom'].astype('int')
        self.data['livingRoom'] = self.data['livingRoom'].astype('int')
        self.data.reset_index(drop=True)
        indes = np.random.choice(self.data.shape[1], replace=True, size=self.data.shape[0])
        self.data = self.data.iloc[indes]


        #self.data.to_csv("China_KSA.csv")

    def split_train_test(self):
        # get the locations
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

        # split the dataset
        s_f =.8
        n_train = math.floor(s_f * self.X.shape[0])
        n_test = math.ceil((1 - s_f) * self.X.shape[0])
        self.X_train = self.X[:n_train]
        self.y_train = self.y[:n_train]
        self.X_test =  self.X[n_train:]
        self.y_test =  self.y[n_train:]

    def scale_data(self):
        #standardization scaler - fit&transform on train, fit only on test
        s_scaler = StandardScaler()
        self.X_train = s_scaler.fit_transform(self.X_train.astype(np.float))
        self.X_test  = s_scaler.transform(self.X_test.astype(np.float))

