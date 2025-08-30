import sys
import os
from src.exception import CustomException
from src.loggers import logging
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            #Train the model
            model.fit(X_train,y_train)
            #Predicting the model
            y_test_preds=model.predict(X_test)
            y_train_preds=model.predict(X_train)

            train_model_score=r2_score(y_train,y_train_preds)
            test_model_score=r2_score(y_test,y_test_preds)

            report[list(models.keys())[i]]=test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)