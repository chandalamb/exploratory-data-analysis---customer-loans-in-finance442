import pandas as pd

# Getting info from dataframe
class DataFrameInfo:

    # Describing datatypes of the column or dataframe 
    def describe_dtypes(self, DataFrame: pd.DataFrame, column_name: str = None): 
        
        # If a column name is provided
        if column_name is not None:
            if column_name not in DataFrame.columns: 
                raise ValueError(f"Column '{column_name}' is not found in the dataframe.") 
            return DataFrame[column_name].dtypes 
        # If a column name is not provided
        else:
            return DataFrame.dtypes
    
    # Finding the median value of the column or dataframe
    def median(self, DataFrame: pd.DataFrame, column_name: str = None): 

        # If a column name is provided
        if column_name is not None:
            if column_name not in DataFrame.columns: 
                raise ValueError(f"Column '{column_name}' not found in the dataframe.") 
            return DataFrame[column_name].median(numeric_only=True) 
        # If a column name is not provided
        else: 
            return DataFrame.median(numeric_only=True) 
    
    # Finding the standard deviation of the column or dataframe
    def standard_deviation(self, DataFrame: pd.DataFrame, column_name: str = None):
        
        # If a column name is provided
        if column_name is not None: 
            if column_name not in DataFrame.columns: 
                raise ValueError(f"Column '{column_name}' not found in the dataframe.")
            return DataFrame[column_name].std(skipna=True, numeric_only=True)
        # If a column name is not provided
        else:
            return DataFrame.std(skipna=True, numeric_only=True) 

    # Finding the mean value of the column or DataFrame
    def mean(self, DataFrame: pd.DataFrame, column_name: str = None): 

        # If a column name is provided
        if column_name is not None:
            if column_name not in DataFrame.columns: 
                raise ValueError(f"Column '{column_name}' not found in the dataframe.")
            return DataFrame[column_name].mean(skipna=True, numeric_only=True) 
        # If a column name is not provided
        else: 
            return DataFrame.mean(skipna=True, numeric_only=True) 
    
    # Finding the number of unique values in a column
    def count_distinct(self, DataFrame: pd.DataFrame, column_name: str):
        return len(DataFrame[column_name].unique())

    # Finding the number of rows and columns in dataframe
    def shape(self, DataFrame: pd.DataFrame):
        print(f'The DataFrame has {DataFrame.shape[1]} columns and {DataFrame.shape[0]} rows.')
        return DataFrame.shape

    # Finding the number of null values in column or dataframe
    def null_count(self, DataFrame: pd.DataFrame, column_name: str = None):
       
        # If a column name is provided
        if column_name is not None:
            if column_name not in DataFrame.columns:
                raise ValueError(f"Column '{column_name}' not found in the dataframe.")
            return DataFrame[column_name].isna().sum()
       # If a column name is not provided
        else:
            return DataFrame.isna().sum()

    # Finding the percentage of null values in a column or dataframe    
    def null_percentage(self, DataFrame: pd.DataFrame, column_name: str = None): 

        # If a column name is provided
        if column_name is not None:
            if column_name not in DataFrame.columns:
                raise ValueError(f"Column '{column_name}' not found in the dataframe.")
            percentage = (DataFrame[column_name].isna().sum())/(len(DataFrame[column_name]))*100 
            return percentage
        # If a column name is not provided
        else:
            percentage = (DataFrame.isna().sum())/(len(DataFrame))*100
            return percentage

    # Getting column names which contain null values    
    def get_null_columns(self, DataFrame: pd.DataFrame, print: bool = False):

        columns_with_null = list(DataFrame.columns[list(DataFrameInfo.null_count(self, DataFrame=DataFrame)>0)])
        if print == True:
            for col in columns_with_null:
                # Get percentage of null values
                print(f'{col}: {round(DataFrameInfo.null_percentage(self, DataFrame=DataFrame, column_name=col),1)} %')
        return columns_with_null
    
    # Get list of numeric columns
    def get_numeric_columns(self, DataFrame: pd.DataFrame):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
        numeric_columns = []
        for column in DataFrame.columns: 
            if DataFrame[column].dtypes in numerics:
                numeric_columns.append(column)
        return numeric_columns

    # Find columns which are over a skewness threshold
    def get_skewed_columns(self, DataFrame: pd.DataFrame, threshold: int):
        numerics_columns = DataFrameInfo.get_numeric_columns(self, DataFrame)
        skewed_columns = []
        for column in numerics_columns:
            if abs(DataFrame[column].skew()) >= threshold: 
                skewed_columns.append(column)
        return skewed_columns

    # Getting a dictionary with columns as keys and skewness as a value
    def get_skewness(self, DataFrame: pd.DataFrame, column_names: list):
        skewness = {}
        for column in column_names: 
            print(f'{column}: {round(DataFrame[column].skew(),2)}')
            skewness[column] = DataFrame[column].skew()
        return skewness
    
    # Calculating the percentage of one column's sum over another column's sum
    def calculate_column_percentage(self, DataFrame: pd.DataFrame, target_column_name: str, total_column_name: str):

        target_column_sum = DataFrame[target_column_name].sum()
        total_column_sum = DataFrame[total_column_name].sum()

        percentage = (target_column_sum / total_column_sum) * 100
        return percentage
    
    # Calculating a percentage 
    def calculate_percentage(self, target, total):
        percentage = (target/total)*100
        return percentage
    
    # Calculating the projection on the total collection over a period
    def calculate_total_collections_over_period(self, DataFrame: pd.DataFrame, period: int):
    
        collections_df = DataFrame.copy() 

        final_payment_date = collections_df['last_payment_date'].max() 

        # Calculating term end
        def calculate_term_end(row):
            # For 36 month terms
            if row['term'] == '36 months':
                return row['issue_date'] + 36
            # For 60 month terms
            elif row['term'] == '60 months':
                return row['issue_date'] + 60

        # Apply the function to create the new 'term_end_date' column
        collections_df['term_end_date'] = collections_df.apply(calculate_term_end, axis=1)

        collections_df['mths_left'] = collections_df['term_end_date'] - final_payment_date # calculate number of months between term end and final payment date.
        collections_df['mths_left'] = collections_df['mths_left'].apply(lambda x: x.n) # Extract integer value from 'mths_left' column.

        collections_df = collections_df[collections_df['mths_left']>0] # filter in only current loans.

        def calculate_collections(row): # Define function to sum collections over projection period.
            if row['mths_left'] >= period: # If months left in term are equal to or greater than projection period.
                return row['instalment'] * period #  projection period * Installments.
            elif row['mths_left'] < period: # If less than projection period months left in term.
                return row['instalment'] * row['mths_left'] # number of months left * installments.

        collections_df['collections_over_period'] = collections_df.apply(calculate_collections, axis=1) # Apply method to each row to get total collections in projected perid.

        collection_sum = collections_df['collections_over_period'].sum()
        total_loan = collections_df['loan_amount'].sum()
        total_loan_left = total_loan - collections_df['total_payment'].sum()

        return {'total_collections': collection_sum, 'total_loan': total_loan, 'total_loan_outstanding': total_loan_left}
    
    # Counting the number of times a value appears in a column
    def count_value_in_column(self, DataFrame: pd.DataFrame, column_name: str, value):
        return len(DataFrame[DataFrame[column_name]==value])
    
   # Getting a list with the cumulative revenue lost each month
    def revenue_lost_by_month(self, DataFrame: pd.DataFrame):
        df = DataFrame.copy()

        df['term_completed'] = (df['last_payment_date'] - df['issue_date'])
        df['term_completed'] = df['term_completed'].apply(lambda x: x.n) 

        def calculate_term_remaining(row):
            # For 36 month terms
            if row['term'] == '36 months':
                return 36 - row['term_completed']
            # For 60 month terms
            elif row['term'] == '60 months':
                return 60 - row['term_completed']

        df['term_left'] = df.apply(calculate_term_remaining, axis=1)
        
        revenue_lost = [] # Empty list
        cumulative_revenue_lost = 0
        for month in range(1, (df['term_left'].max()+1)):
            df = df[df['term_left']>0]
            cumulative_revenue_lost += df['instalment'].sum()
            revenue_lost.append(cumulative_revenue_lost)
            df['term_left'] = df['term_left'] - 1 
        
        return revenue_lost

    # Calculating the total expected revenue from dataframe
    def calculate_total_expected_revenue(self, DataFrame: pd.DataFrame):
        def calculate_total_revenue(row): 
            # For 36 month terms
            if row['term'] == '36 months':
                return 36 * row['instalment']
            # For 60 month terms
            elif row['term'] == '60 months':
                return 60 * row['instalment']

        DataFrame['total_revenue'] = DataFrame.apply(calculate_total_revenue, axis=1)
        total_expected_revenue = DataFrame['total_revenue'].sum()

        return total_expected_revenue
