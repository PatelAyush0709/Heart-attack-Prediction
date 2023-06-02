# import dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
import pickle


# initializing a class
class LogisticRegressionImplementation:
    def __init__(self):
        self.dataframe = None
        self.x = None
        self.y = None
        self.xtrain, self.xtest, self.ytrain, self.ytest = None, None, None, None
        self.model = None
        self.prediction = None
        self.accuracy = None
        self.matrix = None

    # read dataset
    def read_data(self):
        print('\n Read Dataset: ')
        self.dataframe = pd.read_csv('Data/heart.csv')
        print(self.dataframe)

    # dataset information
    def data_information(self):
        print('\n Dataset Information: ')
        print(self.dataframe.info())

    # check null values
    def null_values(self):
        print('\n Check null values in Dataset: ')
        print(self.dataframe.isnull().sum())

    # EDA and preprocessing
    def preprocessing1(self):
        print('\n Correlation between features: ')
        print(self.dataframe.corr())

        print('\n Plotting graphs:')
        plt.figure(figsize=(20, 11))
        sns.heatmap(self.dataframe.corr(), annot=True)
        plt.show()

    # splitting dataset
    def data_splitting(self):
        self.x = self.dataframe.drop(['target'], axis=1)
        self.y = self.dataframe['target']
        print('features column: ', self.x)
        print('Target column: \n', self.y)

    # splitting dataset into train and test
    def train_test(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.x, self.y, test_size=0.3)
        print('length of xtrain: ', len(self.xtrain))
        print('length of xtest: ', len(self.xtest))
        print('length of ytrain: ', len(self.ytrain))
        print('length of ytest: ', len(self.ytest))

    # model selection
    def select_model(self):
        print('--> Model selection: ')
        self.model = LogisticRegression()
        self.model.fit(self.xtrain, self.ytrain)

    # model prediction
    def model_prediction(self):
        print('--> Model prediction')
        self.prediction = self.model.predict(self.xtest)

    # accuracy of the model
    def model_accuracy(self):
        print('Accuracy of the model: ')
        self.accuracy = accuracy_score(self.prediction, self.ytest)
        print(self.accuracy * 100)

    # confusion matrix
    def confusion_matrix1(self):
        print('Confusion matrix: ')
        self.matrix = confusion_matrix(self.prediction, self.ytest)
        print(self.matrix)

    # make pickle file
    def pickle_file(self):
        pkl_file = open("model.pkl", "wb")
        pickle.dump(self.model, pkl_file)
        pkl_file.close()
        print('---> Model is saved in Pickle File and Now its Ready to Use')


if __name__ == '__main__':
    OBJ = LogisticRegressionImplementation()
    OBJ.read_data()
    OBJ.data_information()
    OBJ.null_values()
    OBJ.preprocessing1()
    OBJ.data_splitting()
    OBJ.train_test()
    OBJ.select_model()
    OBJ.model_prediction()
    OBJ.model_accuracy()
    OBJ.confusion_matrix1()
    OBJ.pickle_file()
