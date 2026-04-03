import os 
import sys 

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.fetch_data import FetchData

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """Initializes the DataIngestion instance with the provided DataIngestionConfig.
        Args:
            data_ingestion_config (DataIngestionConfig): The configuration for data ingestion. Defaults to DataIngestionConfig().
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys) #type: ignore
        
    def export_data_into_feature_store(self) -> pd.DataFrame:

        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Exporting data from MongoDB to feature store")
            fetch_data = FetchData()
            
            dfs = {}
            for collection,file_name in self.data_ingestion_config.collections_name.items():
                df = fetch_data.fetch(collection_name=collection)
                logging.info(f"Exported data from {collection} collection successfully")
                dfs[collection] = df

            # merge files on the basis of order_id
            df = pd.merge(dfs["orders"],dfs["payments"], on="order_id", how="left")
            df = pd.merge(df,dfs["reviews"], on="order_id", how="left")
            
            #merge files on the basis of customer_id 
            df = pd.merge(df,dfs["customers"], on="customer_id", how="left") 


            logging.info("Merged dataframe shape: {}".format(df.shape))

            # crete feature store dir if not exist
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Saving merged dataframe to feature store at {feature_store_file_path}")

            df.to_csv(feature_store_file_path, index=False)
            logging.info("Data exported to feature store successfully")
            return df
        except Exception as e:  
            raise MyException(e, sys) #type: ignore


    def split_data_as_train_test(self, df: pd.DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """ 

        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dif_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dif_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path.")
        
        except Exception as e:
            raise MyException(e, sys) #type: ignore

            
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try:
            df = self.export_data_into_feature_store()
            logging.info("Exported data into feature store successfully and got the dataframe")
            
            self.split_data_as_train_test(df=df)
            logging.info("Performed train test split on the dataset")
            
            
            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
            
            data_ingestion_artifact = DataIngestionArtifact(
                training_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) #type: ignore    
    