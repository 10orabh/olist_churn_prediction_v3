import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.constants import MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object ,read_yaml_file
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact


from src.entity.estimator import MyModel


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
             raise MyException(e, sys) # type: ignore



    def get_model_object_and_report(self, train: np.ndarray , test: np.ndarray) -> Tuple[object, object]:

        try:
            logging.info("Training LogisticRegression with specified parameters")
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("Train-test split done.")

            model_config_file = read_yaml_file(file_path=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH)

            best_params = model_config_file['models']['LogisticRegression']['best_params']
            # Initialize LogisticRegression with specified parameters
            model = LogisticRegression(**best_params)

            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # prediction and evaluation 
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # creating matric artifact 
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision , recall_score=recall)
            return model, metric_artifact 
        
        except Exception as e:
            raise MyException(e,sys) #type: ignore
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            acc_score = accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1]))
            if acc_score < self.model_trainer_config.expected_recall:
            
                logging.info(f"Model accuracy is {acc_score} which is less than {self.model_trainer_config.expected_recall}")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e