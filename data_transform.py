import pandas as pd

# Transforming columns within the dataframe
class DataTransform:
    def extract_integer_from_string(self, DataFrame: pd.DataFrame, column_name:str)
        # Extracting digits from the string and puts them into a Int32 data type
        DataFrame[column_name]= DataFrame[column_name].str.extract('(\d+)').astype('Int32')
        return DataFrame
    
    # Replace string with a different string
    def replace_string_text(self, DataFrame: pd.DataFrame, column_name: str, original_string: str, new_string: str):
        DataFrame [column_name] = DataFrame[column_name].str.replace(original_string, new string)
        return DataFrame
    
    # Converting a string formatted date into a period format date
    def convert_date(self, DataFrame: pd. DataFrame, column_name: str)
        # Converting the string into a datetime and convert this to a period of month and year
        DataFrame [column_name] = pd.to_datetime(DataFrame[column_name], errors='coerce').dt.to_period('M')
        return DataFrame
