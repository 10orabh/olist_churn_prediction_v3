import sys 
import src.exception as MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    

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
    
    def run_pipeline(self) -> None:
        """
        Class Name :   TrainPipeline
        Method Name :   run_pipeline
        Description :   This method runs the entire training pipeline
        """

        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise MyException(e, sys) #type: ignore