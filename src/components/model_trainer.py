import os, sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'K-Neighbors': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(random_state=42),
                'CatBoost': CatBoostRegressor(verbose=False, random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42)
            }

            params = {
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'Decision Tree': {
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [2, 3, 5]
                },
                'Linear Regression': {
                    'fit_intercept': [True, False]
                },
                'K-Neighbors': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                },
                'CatBoost': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1],
                    'depth': [4, 6, 8]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            }

            model_report, tuned_models = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = tuned_models[best_model_name]

            print('Best Model:', best_model_name)
            print('Best R2 Score:', best_model_score)
            print('Best Parameters:', best_model.get_params())

            if best_model_score < 0.60:
                raise CustomException('No best model found')

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)