# feature engineering and data tuning
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            num_col = ['writing_score', 'reading_score']
            cat_col = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            logging.info('Standard scaling started for Numeric columns...')
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info('...Standard scaling comleted for Numeric columns...')

            logging.info('Encoding started for Categorical columns...')
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))  # need to check again
                ]
            )
            logging.info('...Encoding comleted for Categorical columns...')

            logging.info(f'Numeric columns - {num_col}')
            logging.info(f'Categorical columns - {cat_col}')
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_col),
                    ('cat_pipeline', cat_pipeline, cat_col)
                ]
            )
            logging.info('...Preprocessor is created...')

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Data transformation is initiated...')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('...Reading train and test data completed...')

            preprocession_obj = self.get_data_transformation_obj()

            target_column_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            input_target_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            input_target_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocession_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocession_obj.transform(input_feature_test_df)

            print("Train feature shape:", input_feature_train_df.shape)
            print("Train target shape:", input_target_train_df.shape)

            print("Test feature shape:", input_feature_test_df.shape)
            print("Test target shape:", input_target_test_df.shape)

            print("Transformed train shape:", input_feature_train_arr.shape)
            print("Transformed test shape:", input_feature_test_arr.shape)
            train_arr = np.c_[
                input_feature_train_arr, np.array(input_target_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(input_target_test_df)
            ]

            logging.info('Saved preprocession object')

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocession_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)