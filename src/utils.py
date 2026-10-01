import os, sys, dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_obj(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        tuned_models = {}

        for name, model in models.items():

            grid = GridSearchCV(
                estimator=model,
                param_grid=params[name],
                scoring='r2',
                cv=5,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            tuned_models[name] = best_model

            train_pred = best_model.predict(X_train)
            test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)

            print(
                f'{name}: Train={train_score:.4f}, '
                f'Test={test_score:.4f}'
            )

            print(f'Best Parameters: {grid.best_params_}')

            report[name] = test_score

        return report, tuned_models

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)