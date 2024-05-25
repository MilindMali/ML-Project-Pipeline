
#handling missing values
#handling outliers
#handling imbalance dataset
#convert categorical column to numerical columns

import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomeException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

#using decoretors you can add extra features to the existng features
@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path=os.path.join("artifacts/data_transformation","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info(" Data transformation started")

            num_feats=['age', 'workclass', 'education_num', 'marital_status', 'occupation',
                       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss','hours_per_week']
        
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            
            # cat_pipeline=Pipeline(
            #     steps=[
            #         ("imputer",SimpleImputer(strategy='mode'))
            #     ]
            # ) since there is no catgorical column anymore

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,num_feats)
            ])

            return preprocessor

        except Exception as e:
            raise CustomeException(e,sys)
        
    def remote_outliers_IQR(self, col, df):
        try:
            q1=df[col].quantile(0.25)
            q3=df[col].quantile(0.75)

            iqr=q3-q1

            upper_limit=q3+(1.5*iqr)
            lower_limit=q1-(1.5*iqr)

            df.loc[(df[col]>upper_limit),col]=upper_limit
            df.loc[(df[col]<lower_limit),col]=lower_limit

            return df

        except Exception as e:
            logging.info("outliers handling code")
            raise CustomeException(e,sys)
        
    def intitate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            num_feats=['age', 'workclass', 'education_num', 'marital_status', 'occupation',
                       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss','hours_per_week']
        
            for col in num_feats:
                self.remote_outliers_IQR(col=col, df=train_data)

            logging.info("outliers capped on our train data")

            for col in num_feats:
                self.remote_outliers_IQR(col=col, df=test_data)

            logging.info("outliers capped on test data")

            preprocess_obj=self.get_data_transformation_object()

            target_col="income"
            drop_column=[target_col]

            logging.info("splitting train data into dependent and independent features")
            input_features_train_data=train_data.drop(drop_column,axis=1)
            target_feature_train_data=train_data[drop_column]

            logging.info("splitting test data into dependent and independent features")
            input_features_test_data=test_data.drop(drop_column,axis=1)
            target_feature_test_data=test_data[drop_column]

            #apply transformation on train data and test data
            input_train_arr=preprocess_obj.fit_transform(input_features_train_data)
            input_test_arr=preprocess_obj.transform(input_features_test_data)

            #apply preprocessor object on our train and test data
            train_array=np.c_[input_train_arr,np.array(target_feature_train_data)]
            test_array=np.c_[input_test_arr,np.array(target_feature_test_data)]

            save_object(file_path=self.data_transformation_config.preprocess_obj_file_path,
                        obj=preprocess_obj)
            
            return (train_array,
                    test_array,
                    self.data_transformation_config.preprocess_obj_file_path)

        except Exception as e:
            raise CustomeException(e,sys)

