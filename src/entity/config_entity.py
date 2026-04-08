import os 
from src.constants import *
from dataclasses import dataclass,field
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass 
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP) 
    timestamp: str = TIMESTAMP 

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

# Data Ingestion related constant start with DATA_INGESTION VAR NAME
@dataclass
class DataIngestionConfig:
    collections_name: dict = field(default_factory=lambda: DATA_INGESTION_COLLECTIONS)
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    

@dataclass 
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransforamationConfig:
    data_trainsformation_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path:str = os.path.join(data_trainsformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TRAIN_FILE_NAME.replace('csv','npy'))
    transformed_test_file_path:str = os.path.join(data_trainsformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TEST_FILE_NAME.replace('csv','npy'))
    transformed_object_file_path:str = os.path.join(data_trainsformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,PREPROCSSING_OBJECT_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_recall = MODEL_TRAINER_EXPECTED_SCORE

@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

@dataclass 
class ChurnPredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME