import sys 
import src.exception as MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact



class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
    

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Class Name :   TrainPipeline
        Method Name :   start_data_ingestion
        Description :   This method starts the data ingestion process and returns a DataIngestionArtifact.
        """
        try:
            logging.info("Starting data ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed successfully and got the training and testing file paths")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) #type: ignore
    
    def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Class Name :   TrainPipeline
        Method Name :   start_data_validation
        Description :   This method starts the data validation process and returns a DataValidationArtifact.
        """
        logging.info("Starting data validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        
        except Exception as e:
            raise MyException(e,sys) #type: ignore

    def run_pipeline(self) -> None:
        """
        Class Name :   TrainPipeline
        Method Name :   run_pipeline
        Description :   This method runs the entire training pipeline
        """

        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        except Exception as e:
            raise MyException(e, sys) #type: ignore
        
    