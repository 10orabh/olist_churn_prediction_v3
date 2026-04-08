import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object, load_numpy_array_data
from src.entity.s3_estimator import Proj1Estimator

@dataclass
class EvaluateModelResponse:
    """
    Dataclass to hold the results of the model evaluation process.
    """
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, 
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Initializes the ModelEvaluation component with required configurations and artifacts.
        """
        try:
            self.model_eval_config = model_eval_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys) from e #type: ignore

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Fetches the currently deployed model from the AWS S3 bucket.
        Returns None if no model is found (e.g., during the first pipeline run).
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)

            # Check if a model already exists in the S3 bucket
            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e, sys)  #type: ignore
        
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compares the newly trained model against the production model using the test dataset.
        Returns an EvaluateModelResponse object containing the decision.
        """
        try:
            logging.info("Loading the fully engineered and scaled test array for evaluation...")
            
            # 1. Load the pre-processed test numpy array (avoids re-doing feature engineering)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            
            # 2. Separate features (X) and target (Y)
            logging.info("Splitting the test array into features (x_scaled) and target (y_true)")
            x_scaled = test_arr[:, :-1]
            y_true = test_arr[:, -1]

            # 3. Retrieve the F1 Score of the newly trained model (calculated by ModelTrainer)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score of the Newly Trained Model: {trained_model_f1_score}")

            best_model_f1_score = 0.0 
            best_model = self.get_best_model()

            # 4. Evaluate the production model if it exists in the S3 bucket
            if best_model is not None:
                logging.info("Computing F1_Score for the OLD Production Model...")
                y_hat_best_model = best_model.predict(x_scaled)
                best_model_f1_score = f1_score(y_true, y_hat_best_model)
                logging.info(f"Production Model F1: {best_model_f1_score} | New Model F1: {trained_model_f1_score}")
            else:
                logging.info("No model found in S3 bucket. This is the first time deployment!")
            
            # 5. Final Decision: Accept the new model if its F1 score is strictly better than the production model
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > best_model_f1_score,
                difference=trained_model_f1_score - best_model_f1_score
            ) 
            logging.info(f"Evaluation Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys) #type: ignore

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Orchestrates the model evaluation process and returns the final evaluation artifact.
        """
        try:
            logging.info("------------------------------------------------------------------------------------------------")
            logging.info("Initialized the Model Evaluation Component.")
            
            # Execute the core evaluation logic
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            # Generate the artifact containing the evaluation results
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_f1=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact created: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise MyException(e, sys) from e  #type: ignore