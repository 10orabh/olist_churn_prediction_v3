import sys
#
from src.entity.config_entity import ChurnPredictorConfig 
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame

class OlistCustomerData:
    def __init__(self,
                 review_score,
                 delivery_days,
                 price,
                 freight_value
                 # ⚠️ IMPORTANT: Yahan aapko apne dataset ke bache hue 24 columns exact usi naam se add karne hain!
                 ):
        """
        Olist Customer Data constructor
        Input: All 28 features required by the trained Random Forest model for churn prediction
        """
        try:
            self.review_score = review_score
            self.delivery_days = delivery_days
            self.price = price
            self.freight_value = freight_value
            # self.next_column_name = next_column_name  <-- Baaki columns yahan assign karein

        except Exception as e:
            raise MyException(e, sys) from e

    def get_customer_input_data_frame(self) -> DataFrame:
        """
        This function converts the OlistCustomerData object into a Pandas DataFrame
        """
        try:
            customer_input_dict = self.get_customer_data_as_dict()
            return DataFrame(customer_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e

    def get_customer_data_as_dict(self):
        """
        This function returns a dictionary from the OlistCustomerData inputs
        """
        logging.info("Entered get_customer_data_as_dict method of OlistCustomerData class")

        try:
            input_data = {
                "review_score": [self.review_score],
                "delivery_days": [self.delivery_days],
                "price": [self.price],
                "freight_value": [self.freight_value]
                # ⚠️ IMPORTANT: Yahan bhi apne baaki saare columns ko dictionary me exact feature names ke sath add karein
            }

            logging.info("Successfully created Olist customer data dictionary")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e


class OlistChurnClassifier:
    def __init__(self, prediction_pipeline_config: ChurnPredictorConfig = ChurnPredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction (contains AWS S3 bucket info & paths)
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> int:
        """
        This method loads the model from S3 (or local path) and makes a churn prediction.
        Returns: Prediction as integer (0 for Loyal Customer, 1 for Churn Risk)
        """
        try:
            logging.info("Entered predict method of OlistChurnClassifier class")
            
            # S3 Estimator ko call karna taaki model (.pkl) load ho sake
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            
            # DataFrame pass karke model se prediction lena
            result = model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)