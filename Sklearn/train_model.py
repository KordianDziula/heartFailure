from data import Data
from model import Model
import joblib

data = Data(file_url = "./heart_failure.csv",
            categorical_features = [ "anaemia",
                                     "diabetes",
                                     "high_blood_pressure",
                                     "sex",
                                     "smoking" ],
            numeric_features = [ "age",
                                 "creatinine_phosphokinase",
                                 "ejection_fraction",
                                 "platelets",
                                 "serum_creatinine",
                                 "serum_sodium",
                                 "time" ],
            target = "DEATH_EVENT").load()

model = Model(data.X_train, data.X_test, 
    data.y_train, data.y_test).train()

print(model.classification_report)
joblib.dump(model.best_estimator, "model.pkl")