import sys
import os 
import pandas as pd 
import numpy as np

from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from src.entity.config_entity import DataTransforamationConfig
from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN, REFRENCE_DATE, THRESHOLD_DAY
from src.logger import logging 
from src.utils.main_utils import read_yaml_file, save_object, save_numpy_array_data
from src.exception import MyException

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTEENN


class DataTransformation:
    
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,data_validation_artifact: DataValidationArtifact,data_transformation_config: DataTransforamationConfig) :
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise MyException(e,sys) #type: ignore
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
       try:
           return pd.read_csv(file_path)
       except Exception as e:
           raise MyException(e, sys) #type: ignore
       

    def get_data_transformer_object(self) -> Pipeline:
        """
        function Name :   get_data_transformer_object
        Description :   Applying Mixed Scaling: 
                        1. MinMaxScaler for Frequency
                        2. StandardScaler for other numeric columns
                        3. Passthrough for Binary features
        Output      :   Pipeline
        On Failure  :   Write an exception log and then raise an exception
        """

        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            standard_scaler_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("std_scaler", StandardScaler())
                ]
            )
            minmax_scaler_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("minmax_scaler", MinMaxScaler())
                ]
            )
            logging.info("Transformer Intialized")

            standard_scaler_features = self._schema_config['std_scaler_columns']
            min_max_scalar_features = self._schema_config['minmax_scaler_columns']
            logging.info("columns loaded from schema")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('StandardScaler', standard_scaler_pipeline, standard_scaler_features),
                    ('MinMaxScaler', minmax_scaler_pipeline, min_max_scalar_features)
            ],
            remainder="passthrough"
            )
            final_pipeline = Pipeline(steps=[("Preprocessor",preprocessor)])

            logging.info("Pipeline is ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline
        except Exception as e:
            raise MyException(e,sys) #type: ignore

    def data_cleaning(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ 
        Method Name :   data_cleaning
        Description :   This method cleans the raw data by removing duplicates 
                        and filtering out incomplete/canceled orders.
        
        Output      :   DataFrame
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info('Data cleaning started')
            
            #  Remove duplicate rows
            logging.info('Removing duplicate rows')
            initial_rows = dataframe.shape[0]
            dataframe = dataframe.drop_duplicates()
            final_rows = dataframe.shape[0]
            logging.info(f"Removed {initial_rows - final_rows} duplicate rows.")
            
            # Filter successful orders
            logging.info("Filtering for 'delivered' orders only")
            dataframe = dataframe[dataframe['order_status'] == 'delivered']

            # fix date columns types
            date_columns = [
                            'order_purchase_timestamp',  
                            'order_delivered_customer_date',
                            'order_estimated_delivery_date'
                            ]

            dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime , errors='coerce')
            
            
            # drop unnecessary columns
            logging.info(f"Drop unnecssary columns: {self._schema_config['drop_columns']}")
            dataframe.drop(columns=self._schema_config['drop_columns'])
            logging.info(f"Data cleaning completed. Final shape: {dataframe.shape}")

            return dataframe

        except Exception as e:
            raise MyException(e, sys) #type: ignore
        
    def perform_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Method Name :   perform_feature_engineering
        Description :   This method applies feature engineering to the raw dataframe, 
                        calculates RFM (Recency, Frequency, Monetary) features, 
                        creates the 'Is_Churn' target variable, and drops unnecessary columns.
        
        Output      :   DataFrame
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Start feature engineering")
        try:
            logging.info("Making feature for model")
            
            df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
            ref_date = pd.to_datetime(REFRENCE_DATE)
            
            df = df.groupby('customer_unique_id').agg(
                Recency=('order_purchase_timestamp', lambda x: (ref_date - x.max()).days),
                Frequency=('order_id', 'nunique'),
                Monetary=('payment_value', 'sum'),
                Average_review_score = ('review_score', 'mean'),
                total_installments= ('payment_installments','sum'),
                Total_Price=('price', 'sum'),
                Total_Freight=('freight_value', 'sum'),
                Avg_delivery_delay = ('delivery_delay', 'mean')
            ).reset_index()
            
            
            # fright_ratio
            df['Freight_ratio'] = df['Total_Freight'] / df['Total_Price']
            df['Freight_ratio'] = df['Freight_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
            df['Avg_delivery_delay'] = df['Avg_delivery_delay'].fillna(0)   
            
           
            # making target (label)
            logging.info("Creating Target Variable: Is_Churn")
            df['Is_churn'] = (df['Recency'] > THRESHOLD_DAY).astype(int)

            # Drop columns
            logging.info(f"Dropping: {self._schema_config['transfromed_drop_columns']}")
            df.drop(columns=self._schema_config["transfromed_drop_columns"],inplace=True)

            logging.info('Feature engineering complete')
            return df
        
        except Exception as e:
             raise MyException(e,sys) # type: ignore

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method orchestrates the entire transformation pipeline:
                        1. Reads Train/Test Data
                        2. Cleans & Engineers Features
                        3. Scales data using Mixed Scaling (Standard + MinMax)
                        4. Saves Preprocessor & Transformed Arrays
        """
        logging.info("Entered initiate_data_transformation method")
        try:
            logging.info("Reading Train and Test datasets")
            train_df = self.read_data(self.data_ingestion_artifact.training_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.testing_file_path)

            # 2. Data Cleaning (Duplicates, Status, Date conversion)
            logging.info("Applying Data Cleaning")
            train_df = self.data_cleaning(train_df)
            test_df = self.data_cleaning(test_df)

            # 3. Feature Engineering (RFM, Freight Ratio, Delay, Churn Label)
            logging.info("Applying Feature Engineering")
            train_df = self.perform_feature_engineering(train_df)
            test_df = self.perform_feature_engineering(test_df)

            # 4. split train test into Features (X) and Target (y)
            logging.info("Splitting Features (X) and Target (y)")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]


            # 4. Create preprocessor pipeline
            logging.info("Creating Preprocessor Object (Standard + MinMax Scalers)")
            preprocessor = self.get_data_transformer_object()

            # 6. Transformation Apply 
            logging.info("Applying Transformation on Train and Test features")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            ) #type: ignore
            
           
            logging.info("SMOTEENN applied to train df.")

            # 7. Concate input scaled feature and target.
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 8. save artifact.
            logging.info("Saving Preprocessor object and Transformed Numpy arrays")
            
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # 9. Response bhejna (Next Component: Model Trainer ke liye)
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("Data Transformation Component completed successfully")
            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys) # type: ignore
        
if __name__ == "__main__":
    try:
        logging.info("🚀 Starting Data Transformation Pipeline via DVC/Terminal...")
    
        # ⚠️ Note: Apne paths check kar lena agar wo alag folder me save hue hain
        from src.entity.artifact_entity import DataValidationArtifact # Agar upar import nahi hai
        
        dummy_ingestion_artifact = DataIngestionArtifact(
            training_file_path="artifacts/data_ingestion/train.csv",
            testing_file_path="artifacts/data_ingestion/test.csv"
        )
        
        dummy_validation_artifact = DataValidationArtifact(
            validation_status=True,
            message="Data is valid for transformation"
        )
        
        # 2. Transformation ka apna config load karna
        transformation_config = DataTransforamationConfig()
        
        # 3. Object banaiye aur run kijiye
        data_transformation = DataTransformation(
            data_ingestion_artifact=dummy_ingestion_artifact,
            data_validation_artifact=dummy_validation_artifact,
            data_transformation_config=transformation_config
        )
        
        # 4. Main transformation method trigger kijiye
        transformation_artifact = data_transformation.initiate_data_transformation()
        
        logging.info(f"✅ Data Transformation Completed! Artifacts: {transformation_artifact}")
        print("✅ Data Transformation Successful! Preprocessor and Numpy Arrays saved.")
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed at Data Transformation stage: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1) # DVC ko batao ki fail ho gaya hai