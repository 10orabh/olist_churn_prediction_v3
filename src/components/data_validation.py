import json
import os 
import sys 

import pandas as pd 
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self,data_validation_config: DataValidationConfig ,data_ingestion_artifact: DataIngestionArtifact ) -> None:
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys) #type: ignore
        
    
    def validate_number_of_column(self,dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"is required column present: [{status}]")
            return status
        except Exception as e:
            raise MyException(e,sys) #type: ignore
    
    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            
            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

                
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            
            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns) else True
        
        except Exception as e:
            raise MyException(e,sys) from e  #type: ignore
        

    @ staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise MyException(e,sys) #type: ignore


    def initiate_data_validation(self) -> DataValidationArtifact:

        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """    

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df,test_df = (
            DataValidation.read_data(file_path=self.data_ingestion_artifact.training_file_path),
            DataValidation.read_data(file_path=self.data_ingestion_artifact.testing_file_path)
                                )
            

            # check the column len of train and test column
            train_status = self.validate_number_of_column(train_df)

            if not train_status:
                validation_error_msg += "Columns are missing in train dataframe./n"
            else:
                logging.info(f"All require columns are present in train dataframe: {train_status}")

            test_status = self.validate_number_of_column(test_df)

            if not test_status:
                validation_error_msg += "Columns are missing in test dataframe./n"
            else:
                logging.info(f"All require columns are present in test dataframe: {test_status}")


            # check if all the numerical and categorical column are present in the train and test dataframe.
        
            train_status = self.is_column_exist(train_df)

            if not train_status:
                validation_error_msg += "Columns are missing in train dataframe./n"
            else:
                logging.info(f"All Categorical and Numerical columns are present in train dataframe: {train_status}")

            
            test_status = self.is_column_exist(test_df)

            if not test_status:
                validation_error_msg += "Columns are missing in test dataframe./n"
            else:
                logging.info(f"All Categorical and Numerical columns are present in test dataframe: {test_status}")


            validate_status = len(validation_error_msg) == 0 

            # aa the configuration to data validation artifact 
            data_validation_artifact = DataValidationArtifact(
                validation_status=validate_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)

            os.makedirs(report_dir,exist_ok=True)

            validation_report = {
                "validation_status": validate_status,
                "message": validation_error_msg.strip()
            }
            
            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
              json.dump(validation_report, report_file, indent=4)
            
            
            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e #type: ignore
        


