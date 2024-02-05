import numpy as np
import pandas as pd
from scipy import stats
from dataframe_info import DataFrameInfo as info
from plotter import Plotter as plotter


np.random.seed(123)

# Transforming dataframe to edit columns with missing data
class DataFrameTransform:

    # Removing columns which contain null/missing values
    def remove_null_columns(self, DataFrame: pd.DataFrame, column_name):
        DataFrame.drop(column_name, axis=1, inplace=True)
        return DataFrame

    # Removing rows in dataframe where data points are null in a specified column
    def remove_null_rows(self, DataFrame: pd.DataFrame, column_name):
        DataFrame.dropna(subset=column_name, inplace=True)
        return DataFrame
    
    # Replace null values with the median value
    def fill_median(self, DataFrame: pd.DataFrame, column_name):
        DataFrame[column_name].fillna(DataFrame[column_name].median(numeric_only=True), inplace=True)
        return DataFrame
    
    # Replace null values with the mean value
    def fill_mean(self, DataFrame: pd.DataFrame, column_name):
        DataFrame[column_name].fillna(DataFrame[column_name].mean(numeric_only=True, skipna=True), inplace=True)
        return DataFrame
    
    # Attribute null values in a numerical column based on a linear regression model
    def linear_regression_fill(self, DataFrame: pd.DataFrame, column_to_fill: str, training_features: list = None, score: bool = False, check_distribution: bool = False):
        
        # Plotting histogram
        if check_distribution == True:
            print(f'\n({column_to_fill}) Initial Distribution:\n')
            plotter.histogram(self, DataFrame, column_to_fill)

        if training_features == None: # If no training features are provided
            x = DataFrame.drop(info.get_null_columns(self, DataFrame), axis=1) 
        else:  # If training features are provided
            x = DataFrame[training_features] 
        y = DataFrame[column_to_fill]

        # Encodeing string columns to numeric type 
        object_columns = x.select_dtypes(include=['object']).columns.tolist() 
        x[object_columns] = x[object_columns].astype('category') 
        x[object_columns] = x[object_columns].apply(lambda x: x.cat.codes) 

        # Encoding date columns to numeric type
        date_columns = x.select_dtypes(include=['period[M]']).columns.tolist() 
        x[date_columns] = x[date_columns].astype('category') 
        x[date_columns] = x[date_columns].apply(lambda x: x.cat.codes) 


        x_train = x[~y.isna()] 
        y_train = y[~y.isna()] 

        x_test = x[y.isna()] # Testing input data

        # Train Linear Regression Model:
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Run model and impute null values with predicted values
        prediction = model.predict(x_test)
        DataFrame[column_to_fill].loc[y.isna()] = prediction
        
        if check_distribution == True:
            print(f'\n({column_to_fill}) Final Distribution:\n')
            plotter.histogram(self, DataFrame, column_to_fill) # Plots histogram to display distribution

        if score == True:
            print(f'\nScore: {round(model.score(x_train, y_train),2)}') # Provides an accuracy score for the model

        return DataFrame

    # Applying Box-Cox transformation to normalise a column 
    def box_cox_transform(self, DataFrame: pd.DataFrame, column_name: str):
        boxcox_column = stats.boxcox(DataFrame[column_name])
        boxcox_column = pd.Series(boxcox_column[0])
        return boxcox_column

    # Applying Yeo-Johnson transformation to normalise a column
    def yeo_johnson_transform(self, DataFrame: pd.DataFrame, column_name: str):
        yeojohnson_column = stats.yeojohnson(DataFrame[column_name])
        yeojohnson_column = pd.Series(yeojohnson_column[0])
        return yeojohnson_column

    # Removing rows based on the 'z score' of values in a column
    def drop_outlier_rows(self, DataFrame: pd.DataFrame, column_name: str, z_score_threshold: int):
        mean = np.mean(DataFrame[column_name]) # Finding the mean of the column
        std = np.std(DataFrame[column_name]) # Finding the standard deviation of the column
        z_scores = (DataFrame[column_name] - mean) / std # Finding the 'z score' for each value in the column
        abs_z_scores = pd.Series(abs(z_scores)) # Create series with the values of the 'z_score'
        mask = abs_z_scores < z_score_threshold
        DataFrame = DataFrame[mask] # Keep only rows where the 'z score' is below the threshold       
        return DataFrame
    
    # Imputing null values in a categorical column
    def support_vector_machine_fill(self, DataFrame: pd.DataFrame, column_to_fill: str, training_features: list = None, score: bool = False, check_distribution: bool = False):

        if check_distribution == True:
            initial_distribution = DataFrame[column_to_fill].value_counts(normalize=True)

        if training_features == None: 
            x = DataFrame.drop(info.get_null_columns(self, DataFrame), axis=1)
        else: 
            x = DataFrame[training_features] 
        y = DataFrame[column_to_fill] 

        # Encode string columns to numeric type
        object_columns = x.select_dtypes(include=['object']).columns.tolist() 
        x[object_columns] = x[object_columns].astype('category') 
        x[object_columns] = x[object_columns].apply(lambda x: x.cat.codes) 

        # Encode date columns to numeric type
        date_columns = x.select_dtypes(include=['period[M]']).columns.tolist() 
        x[date_columns] = x[date_columns].astype('category') 
        x[date_columns] = x[date_columns].apply(lambda x: x.cat.codes)

        scaler = RobustScaler()
        transformer = scaler.fit(x)
        transformer.transform(x)

        # Data Split:
        sample_size = (DataFrame[column_to_fill].isna().sum()) * 4 
        if sample_size < 10000:
            sample_size = 10000 
        x_train = x[~y.isna()].sample(sample_size, random_state=123)
        y_train = y[x.index.isin(x_train.index)]

        x_test = x[y.isna()]

        model = SVC()
        model.fit(x_train, y_train)

        # Run model and impute null values with predicted values:
        prediction = model.predict(x_test)
        DataFrame[column_to_fill].loc[y.isna()] = prediction 

        if check_distribution == True:
            final_distribution = DataFrame[column_to_fill].value_counts(normalize=True)
            distribution_df = pd.DataFrame({'Before': round(initial_distribution, 3),'After': round(final_distribution, 3)})
            print('Distribution: Normalised Value Count')
            print(distribution_df)
        
        if score == True:
            print(f'\nScore: {round(model.score(x_train, y_train),2)}')
        
        return DataFrame
    
