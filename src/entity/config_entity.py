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
    