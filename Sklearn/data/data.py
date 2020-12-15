import pandas as pd
from sklearn.model_selection import train_test_split

class Data():
    def __init__(self, file_url: str, categorical_features: list, numeric_features: list, target: str):
        self.file_url = file_url
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.target = target
    
    def load(self):
        df = pd.read_csv(self.file_url)
        df[self.numeric_features] = df[self.numeric_features].astype("float")
        df[self.categorical_features] = df[self.categorical_features].astype("category")
        
        columns = self.categorical_features + self.numeric_features
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(df[columns], 
                             df[self.target], 
                             test_size = 0.2, 
                             stratify = df[self.target])
        
        return self
