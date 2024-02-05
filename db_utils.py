import yaml
import sqlalchemy
import pandas as pd
from sqlalchemy import create_engine


# Extracting the remote database connection credentials from the yaml file and returning the data dictionary contained within
def extract_credentials ():
    with open ('credentials.yaml','r') as file:
        return yaml.safe_load(file)

# Store dictionary into a variable
credentials_dict = extract_credentials()


# Using class to connect to the RDS database and extract data on loan payments
class RDSDatabaseConnector:
    def __init__(self, credentials_dict):
        self.credentials_dict = credentials_dict
   
   # Initialising SQLAlchemy engine
    def create_engine (self):
        self.engine = create_engine(f"postgresql+psycopg2://{self.credentials_dict['RDS_USER']}:{self.credentials_dict['RDS_PASSWORD']}@{self.credentials_dict['RDS_HOST']}:{self.credentials_dict['RDS_PORT']}/{self.credentials_dict['RDS_DATABASE']}")

    # Connecting with the database and creating a pandas df from 'loan payment' table
    def extract_data(self):
        with self.engine.connect() as connection:
            self.loan_payments_df = pd.read_sql_table('loan_payments', self.engine)
            return self.loan_payments_df

    # Saving data into a csv file
    def save_data(loans_df: pd.DataFrame):
        with open ('loan_payments.csv', 'w') as file:
            loans_df.to_csv(file, encoding = 'utf-8', index = false)

if __name__ == '__main__':
    connector = RDSDatabaseConnector (credentials_dict)
    connector.create_engine()
    extracted_data_frame: pd.DataFrame = connector.extract_data
    save_data(extracted_data_frame)
