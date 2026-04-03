import sys 
import pandas as pd 
import numpy as np 
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.exception import MyException
from src.constants import DATABASE_NAME

class FetchData:
    """
    FetchData is responsible for fetching data from the MongoDB database.

    Methods:
    -------
    fetch_data(collection_name: str, query: Optional[dict] = None) -> pd.DataFrame
        Fetches data from the specified MongoDB collection based on the provided query and returns it as a pandas DataFrame.
    """

    def __init__(self):
        """
        Initializes the FetchData instance by creating a MongoDBClient to connect to the database.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)  # Create an instance of MongoDBClient to manage database connection
        except Exception as e:
            raise MyException(e, sys) #type: ignore

    def fetch(self, collection_name: str, query: Optional[dict] = None,database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Fetches data from the specified MongoDB collection based on the provided query and returns it as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to fetch data from.
        query : dict, optional
            A dictionary representing the MongoDB query to filter the data. If None, all documents will be fetched.
        database_name : str, optional
            The name of the MongoDB database to fetch data from. If None, the default database specified in the MongoDBClient will be used.

        Returns:
        -------
        pd.DataFrame
            A pandas DataFrame containing the fetched data from the MongoDB collection.

        Raises:
        ------
        MyException
            If there is an issue fetching data from MongoDB or if the specified collection does not exist.
        """
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                # Access the specified collection in the connected database
                collection = self.mongo_client.database[database_name][collection_name]

            # Execute the query to fetch data; if no query is provided, fetch all documents
            if query is None:
                query = {}
            cursor = collection.find(query)
            
            # Convert the cursor to a list of documents and then to a pandas DataFrame
            data_list = list(cursor)
            df = pd.DataFrame(data_list)
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            
            df.replace({"na": np.nan}, inplace=True)
            
            return df
        
        except Exception as e:
            # Raise a custom exception with traceback details if fetching fails
            raise MyException(e, sys) #type: ignore