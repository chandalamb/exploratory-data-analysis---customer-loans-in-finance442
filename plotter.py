from matplotlib import pyplot
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

# Plotting visualisations of the data
class Plotter:
    
    # Plotting a histogram for data in a column
    def histogram(self, DataFrame: pd.DataFrame, column_name: str):
        fig = px.histogram(DataFrame, column_name)
        return fig.show()
    
    # Plotting a histogram for data in a column, identifying skewness 
    def skewness_histogram(self, DataFrame: pd.DataFrame, column_name: str):
        histogram = sns.histplot(DataFrame[column_name],label="Skewness: %.2f"%(DataFrame[column_name].skew()) )
        histogram.legend()
        return histogram

    # Plotting a matrix to display null data points
    def missing_matrix(self, DataFrame: pd.DataFrame):
        return msno.matrix(DataFrame)
    
    # Returning Quantile-Quantile (Q-Q) plot
    def qqplot(self, DataFrame: pd.DataFrame, column_name: str):
        qq_plot = qqplot(DataFrame[column_name] , scale=1 ,line='q') 
        return pyplot.show()
    
    # Returning a Facet Grid with histograms with the distrubution of columns
    def facet_grid_histogram(self, DataFrame: pd.DataFrame, column_names: list):
        melted_df = pd.melt(DataFrame, value_vars=column_names) 
        facet_grid = sns.FacetGrid(melted_df, col="variable",  col_wrap=3, sharex=False, sharey=False) 
        facet_grid = facet_grid.map(sns.histplot, "value", kde=True)
        return facet_grid
    
    # Returning a Facet Grid with box-plots of a list of columns
    def facet_grid_box_plot(self, DataFrame: pd.DataFrame, column_names: list):
        melted_df = pd.melt(DataFrame, value_vars=column_names) 
        facet_grid = sns.FacetGrid(melted_df, col="variable",  col_wrap=3, sharex=False, sharey=False)
        facet_grid = facet_grid.map(sns.boxplot, "value", flierprops=dict(marker='x', markeredgecolor='red'))
        return facet_grid
    
    # Comparing transformations on skewness using subplots 
    def compare_skewness_transformations(self, DataFrame: pd.DataFrame, column_name: str):

        transformed_df = DataFrame.copy()

        # Applying transformations and create new column with transformed data
        transformed_df['log_transformed'] = DataFrame[column_name].map(lambda x: np.log(x) if x > 0 else 0)
        if (DataFrame[column_name] <= 0).values.any() == False: 
            transformed_df['box_cox'] = pd.Series(stats.boxcox(DataFrame[column_name])[0]).values # Perform box-cox transformation and add values as new column
        transformed_df['yeo-johnson'] = pd.Series(stats.yeojohnson(DataFrame[column_name])[0]).values # Perform yeo-johnson transformation and add values as new column

        # Creating a figure and subplots
        if (DataFrame[column_name] <= 0).values.any() == False: 
            fig, axes = pyplot.subplots(nrows=2, ncols=4, figsize=(16, 8)) 
        else: 
            fig, axes = pyplot.subplots(nrows=2, ncols=3, figsize=(16, 8))

        # Setting titles of subplots
        axes[0, 0].set_title('Original Histogram')
        axes[1, 0].set_title('Original Q-Q Plot')
        axes[0, 1].set_title('Log Transformed Histogram')
        axes[1, 1].set_title('Log Transformed Q-Q Plot')
        if (DataFrame[column_name] <= 0).values.any() == False:        
            axes[0, 2].set_title('Box-Cox Transformed Histogram')
            axes[1, 2].set_title('Box-Cox Transformed Q-Q Plot')
            axes[0, 3].set_title('Yeo-Johnson Transformed Histogram')
            axes[1, 3].set_title('Yeo-Johnson Transformed Q-Q Plot')
        else:
            axes[0, 2].set_title('Yeo-Johnson Transformed Histogram')
            axes[1, 2].set_title('Yeo-Johnson Transformed Q-Q Plot')

        # Add Histograms to subplots
        sns.histplot(DataFrame[column_name], kde=True, ax=axes[0, 0]) 
        axes[0, 0].text(0.5, 0.95, f'Skewness: {DataFrame[column_name].skew():.2f}', ha='center', va='top', transform=axes[0, 0].transAxes) 
        sns.histplot(transformed_df['log_transformed'], kde=True, ax=axes[0, 1]) 
        axes[0, 1].text(0.5, 0.95, f'Skewness: {transformed_df["log_transformed"].skew():.2f}', ha='center', va='top', transform=axes[0, 1].transAxes) 
        if (DataFrame[column_name] <= 0).values.any() == False: 
            sns.histplot(transformed_df['box_cox'], kde=True, ax=axes[0, 2]) 
            axes[0, 2].text(0.5, 0.95, f'Skewness: {transformed_df["box_cox"].skew():.2f}', ha='center', va='top', transform=axes[0, 2].transAxes) 
            sns.histplot(transformed_df['yeo-johnson'], kde=True, ax=axes[0, 3]) 
            axes[0, 3].text(0.5, 0.95, f'Skewness: {transformed_df["yeo-johnson"].skew():.2f}', ha='center', va='top', transform=axes[0, 3].transAxes) 
        else: 
            sns.histplot(transformed_df['yeo-johnson'], kde=True, ax=axes[0, 2]) 
            axes[0, 2].text(0.5, 0.95, f'Skewness: {transformed_df["yeo-johnson"].skew():.2f}', ha='center', va='top', transform=axes[0, 2].transAxes)

        # Add Q-Q plots to subplots
        stats.probplot(DataFrame[column_name], plot=axes[1, 0]) 
        stats.probplot(transformed_df['log_transformed'], plot=axes[1, 1])
        if (DataFrame[column_name] <= 0).values.any() == False: 
            stats.probplot(transformed_df['box_cox'], plot=axes[1, 2])
            stats.probplot(transformed_df['yeo-johnson'], plot=axes[1, 3])
        else: 
            stats.probplot(transformed_df['yeo-johnson'], plot=axes[1, 2]) 

        pyplot.suptitle(column_name, fontsize='xx-large') 
        pyplot.tight_layout()
        return pyplot.show()
    
    # Returning subplots to show the effects of skewness transformation
    def before_after_skewness_transformation(self, DataFrame: pd.DataFrame, column_name: str):

        # Importing original dataframe column data into seperate dataframe
        df_original = pd.read_csv('loan_payments_versions/loan_payments_post_null_imputation.csv')

        fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16, 8))

        # Creating Q-Q Sub-Plots
        stats.probplot(df_original[column_name], plot=axes[0]) 
        stats.probplot(DataFrame[column_name], plot=axes[1])

        # Adding skewness
        axes[0].text(0.5, 0.95, f'Skewness: {df_original[column_name].skew():.2f}', ha='center', va='top', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.95, f'Skewness: {DataFrame[column_name].skew():.2f}', ha='center', va='top', transform=axes[1].transAxes) 

        # Adding Sub-Plot titles
        axes[0].set_title('Q-Q Plot: Before', fontsize='x-large')
        axes[1].set_title('Q-Q Plot: After', fontsize='x-large')

        pyplot.suptitle(column_name, fontsize='xx-large') 
        return pyplot.show()
    
    # Creating box-plots of a column
    def box_plot(self, DataFrame: pd.DataFrame, column_name: str):
        sns.boxplot(x=column_name, data = DataFrame, flierprops=dict(marker='x', markeredgecolor='red')) 
        return pyplot.show()
    
    # Returning subplots to show the effect of a outlier removal transformation 
    def before_after_outlier_removal(self, DataFrame: pd.DataFrame, column_name: str):

        # Importing original dataframe column data into seperate dataframe
        df_original = pd.read_csv('loan_payments_versions/loan_payments_post_skewness_correction.csv')

        fig, axes = pyplot.subplots(nrows=2, ncols=2, figsize=(16, 8))

        # Add box-plots
        sns.boxplot(x=column_name, data = df_original, flierprops=dict(marker='x', markeredgecolor='red'), ax=axes[0, 0]) # Original
        sns.boxplot(x=column_name, data = DataFrame, flierprops=dict(marker='x', markeredgecolor='red'), ax=axes[0, 1]) # Transformed

        # Add histograms
        sns.histplot(df_original[column_name], ax=axes[1, 0]) # Original
        sns.histplot(DataFrame[column_name], ax=axes[1, 1]) # Transformed

        # Set sub-plot titles
        axes[0, 0].set_title('Box Plot: Before')
        axes[0, 1].set_title('Box Plot: After')
        axes[1, 0].set_title('Histogram: Before')
        axes[1, 1].set_title('Histogram: After')

        pyplot.suptitle(column_name, fontsize='xx-large') 
        pyplot.subplots_adjust(hspace=0.3)
        return pyplot.show()
    
    # Producing a correlation matrix heatmap
    def correlation_matrix(self, DataFrame: pd.DataFrame):
        for column in DataFrame.columns: 
            if DataFrame[column].dtype not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                raise ValueError(f"The '{column}' column is not numerical datatype.") 

        corr = DataFrame.corr() 

        mask = np.zeros_like(corr, dtype=np.bool_) 
        mask[np.triu_indices_from(mask)] = True

        cmap = sns.color_palette("viridis", as_cmap=True) 

        pyplot.figure(figsize=(14, 12))

        # Generate heatmap
        sns.heatmap(corr, mask=mask, square=True, linewidths=.5, annot=True, cmap=cmap, fmt=".2f")
        pyplot.yticks(rotation=0)
        pyplot.title('Correlation Matrix of all Numerical Variables')
        return pyplot.show()

    # Creating a bar chart plot of categorical data
    def bar_chart(self, independant_categories: list, dependant_variables: list, title: str=None, y_label: str=None, x_label: str=None):
        
        pyplot.figure(figsize=(16, 8))
        sns.barplot(x=independant_categories, y=dependant_variables)
        if y_label != None:
            pyplot.ylabel(y_label)
        if x_label != None: 
            pyplot.xlabel(x_label)
        if title != None: 
            pyplot.title(title)
        return pyplot.show()

    # Creating a pie chart plot of categorical data
    def pie_chart(self, labels: list, sizes: list, title: str=None):

        pyplot.pie(sizes, labels=labels, colors=['#66b3ff', '#ffff99', '#00FF00'], autopct='%1.1f%%', startangle=180)
        if title != None:
            pyplot.title(title)
        pyplot.show()
    
    # Creating two pie chart subplots
    def two_pie_charts(self, labels_1: list, sizes_1: list, labels_2: list, sizes_2: list, plot_title: str=None, title_1: str=None, title_2: str=None):

        fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(16, 8)) 

        if title_1 != None: 
            axes[0].set_title(title_1)
        if title_2 != None: 
            axes[1].set_title(title_2)

        axes[0].pie(sizes_1, labels=labels_1, colors=['#66b3ff', '#ffff99', '#00FF00'], autopct='%1.1f%%', startangle=90) 
        axes[1].pie(sizes_2, labels=labels_2, colors=['#66b3ff', '#ffff99', '#00FF00'], autopct='%1.1f%%', startangle=90)

        if plot_title != None:
            pyplot.suptitle(plot_title, fontsize='xx-large')

        return pyplot.show()

    # Producing a grid with bar chart subplots
    def two_bar_charts(self, independant_categories_1: list, dependant_variables_1: list, independant_categories_2: list, dependant_variables_2: list, plot_title: str=None, title_1: str=None, title_2: str=None, y_label_1: str=None, x_label_1: str=None, y_label_2: str=None, x_label_2: str=None):
        
        fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 6))

        if title_1 != None:
            axes[0].set_title(title_1)
        if title_2 != None:
            axes[1].set_title(title_2)

        sns.barplot(x=independant_categories_1, y=dependant_variables_1, ax=axes[0])
        sns.barplot(x=independant_categories_2, y=dependant_variables_2, ax=axes[1])

        if y_label_1 != None: 
            axes[0].set_ylabel(y_label_1)
        if x_label_1 != None:
            axes[0].set_xlabel(x_label_1)

        if y_label_2 != None:
            axes[1].set_ylabel(y_label_2)
        if x_label_2 != None:
            axes[1].set_xlabel(x_label_2)

        if plot_title != None:
            pyplot.suptitle(plot_title, fontsize='xx-large')

        return pyplot.show()
    
    # Creating a discrete popultaion distribution bar plot for a column in a dataframe
    def discrete_population_distribution(self, DataFrame: pd.DataFrame, column_name: str, title: str=None, y_label: str=None, x_label: str=None):

        probabilities = DataFrame[column_name].value_counts(normalize=True)

        pyplot.figure(figsize=(16, 8))
        pyplot.rc("axes.spines", top=False, right=False)
        sns.barplot(y=probabilities.index, x=probabilities.values, color='b')

        if y_label != None: # If there is a 'y_label'
            pyplot.ylabel(y_label)
        if x_label != None: # If there is a 'x_label'
            pyplot.xlabel(x_label)
        if title != None: # If there is a 'title'
            pyplot.title(title)
        return pyplot.show()

    # Creating a scatter plot to show the relationship between two variables
    def scatter_plot(self, DataFrame: pd.DataFrame, x_variable: str, y_variable: str, title: str=None):
       
        pyplot.figure(figsize=(16, 8))
        sns.scatterplot(data=DataFrame, x=x_variable, y=y_variable) # Generate scatter plot between two variables

        if title != None:
            pyplot.title(title)
        return pyplot.show()
    
    # Creating a pairplot of scatter subplots of pairs of variables
    def pair_plot(self, DataFrame: pd.DataFrame):
        return sns.pairplot(DataFrame)

    # Creating a pie chart for a columnn
    def column_pie_chart(self, DataFrame: pd.DataFrame, column_name: str, title: str=None, y_label: str=None, x_label: str=None):

        probabilities = DataFrame[column_name].value_counts(normalize=True)

        pyplot.figure(figsize=(16, 8))
        pyplot.pie(list(probabilities.values), labels=list(probabilities.index), colors=['#66b3ff', '#ffff99', '#00FF00'], autopct='%1.1f%%', startangle=180)

        if y_label != None: 
            pyplot.ylabel(y_label)
        if x_label != None:
            pyplot.xlabel(x_label)
        if title != None:
            pyplot.title(title)
        return pyplot.show()

    # Showing the probabilty of discrete values
    def discrete_value_risk_comparison(self, DataFrame: pd.DataFrame, column_name: str):

        # Defining DataFrames that contain subsets of loan status
        df = DataFrame
        paid_df = df[df['loan_status'] == 'Fully Paid']
        charged_default_df = df[df['loan_status'].isin(['Charged Off','Default'])] 
        risky_df = df[df['loan_status'].isin(['Late (31-120 days)','In Grace Period', 'Late (16-30 days)'])]

        # Getting proportions of discrete values in column, only selecting the top 8.
        probabilities = DataFrame[column_name].value_counts(normalize=True).head(8)
        paid_probabilities = paid_df[column_name].value_counts(normalize=True).head(8)
        charged_default_probabilities = charged_default_df[column_name].value_counts(normalize=True).head(8)
        risky_probabilities = risky_df[column_name].value_counts(normalize=True).head(8)

        # Generating main plot
        fig, axes = pyplot.subplots(nrows=2, ncols=4, figsize=(16, 8))

        # Setting titles
        axes[0, 0].set_title('All Loans')
        axes[0, 1].set_title('Fully Paid Loans')
        axes[0, 2].set_title('Charged off and Default Loans')
        axes[0, 3].set_title('Risky Loans')

        colour_palette = ['#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#ffff99', '#1f78b4']

        # Generating subplot pie charts
        axes[0, 0].pie(list(probabilities.values), labels=list(probabilities.index), colors=colour_palette, autopct='%1.1f%%', startangle=90)
        axes[0, 1].pie(list(paid_probabilities.values), labels=list(paid_probabilities.index), colors=colour_palette, autopct='%1.1f%%', startangle=90)
        axes[0, 2].pie(list(charged_default_probabilities.values), labels=list(charged_default_probabilities.index), colors=colour_palette, autopct='%1.1f%%', startangle=90)
        axes[0, 3].pie(list(risky_probabilities.values), labels=list(risky_probabilities.index), colors=colour_palette, autopct='%1.1f%%', startangle=90)

        # Removing top and right spine for bottom plots
        axes[1, 0].spines["top"].set_visible(False)
        axes[1, 0].spines["right"].set_visible(False)
        axes[1, 1].spines["top"].set_visible(False)
        axes[1, 1].spines["right"].set_visible(False)
        axes[1, 2].spines["top"].set_visible(False)
        axes[1, 2].spines["right"].set_visible(False)
        axes[1, 3].spines["top"].set_visible(False)
        axes[1, 3].spines["right"].set_visible(False)

        # Generating bar plots
        sns.barplot(y=probabilities.index, x=probabilities.values, color='#a6cee3', ax=axes[1,0])
        sns.barplot(y=paid_probabilities.index, x=paid_probabilities.values, color='#a6cee3', ax=axes[1,1])
        sns.barplot(y=charged_default_probabilities.index, x=charged_default_probabilities.values, color='#a6cee3', ax=axes[1,2])
        sns.barplot(y=risky_probabilities.index, x=risky_probabilities.values, color='#a6cee3', ax=axes[1,3])

        pyplot.suptitle(column_name, fontsize='xx-large') 
        pyplot.tight_layout()

        return pyplot.show()

    # Showing the distrubtion and averages of continous vales in the dataframe
    def continuous_value_risk_comparison(self, DataFrame: pd.DataFrame, column_name: str, z_score_threshold: float=3):
        
        def drop_outliers(Data_Frame: pd.DataFrame, column_name: str, z_score_threshold: float):
            mean = np.mean(Data_Frame[column_name])
            std = np.std(Data_Frame[column_name])
            z_scores = (Data_Frame[column_name] - mean) / std
            abs_z_scores = pd.Series(abs(z_scores))
            mask = abs_z_scores < z_score_threshold
            Data_Frame = Data_Frame[mask]      
            return Data_Frame

        # Defining DataFrames that contain subsets of loan status
        df = drop_outliers(DataFrame, column_name, z_score_threshold)
        paid_df = df[df['loan_status'] == 'Fully Paid']
        charged_default_df = df[df['loan_status'].isin(['Charged Off','Default'])] 
        risky_df = df[df['loan_status'].isin(['Late (31-120 days)','In Grace Period', 'Late (16-30 days)'])]

        # Generating main plot
        fig, axes = pyplot.subplots(nrows=2, ncols=4, figsize=(20, 10))

        # Setting titles
        axes[0, 0].set_title(f'All Loans\nMean: {round(df[column_name].mean(),1)}')
        axes[0, 1].set_title(f'Fully Paid Loans\nMean: {round(paid_df[column_name].mean(),1)}')
        axes[0, 2].set_title(f'Charged off and Default Loans\nMean: {round(charged_default_df[column_name].mean(),1)}')
        axes[0, 3].set_title(f'Risky Loans\nMean: {round(risky_df[column_name].mean(),1)}')

        colour_palette = ['#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#ffff99', '#1f78b4']

        # Generating subplot histograms
        sns.histplot(data=df, x=column_name, kde=True, color='#a6cee3', ax=axes[0, 0])
        sns.histplot(data=paid_df, x=column_name, kde=True, color='#a6cee3', ax=axes[0, 1])
        sns.histplot(data=charged_default_df, x=column_name, kde=True, color='#a6cee3', ax=axes[0, 2])
        sns.histplot(data=risky_df, x=column_name, kde=True, color='#a6cee3', ax=axes[0, 3])
        
        # Adding vertical mean lines
        axes[0, 0].axvline(df[column_name].mean(), color='blue', linestyle='dashed', linewidth=1.5, label='Mean')
        axes[0, 1].axvline(paid_df[column_name].mean(), color='blue', linestyle='dashed', linewidth=1.5, label='Mean')
        axes[0, 2].axvline(charged_default_df[column_name].mean(), color='blue', linestyle='dashed', linewidth=1.5, label='Mean')
        axes[0, 3].axvline(risky_df[column_name].mean(), color='blue', linestyle='dashed', linewidth=1.5, label='Mean')

        # Removing spine from histograms
        sns.despine(ax=axes[0, 0])
        sns.despine(ax=axes[0, 1])
        sns.despine(ax=axes[0, 2])
        sns.despine(ax=axes[0, 3])

        # Generating violin plots
        sns.violinplot(data=df, y=column_name, color='#fb9a99', ax=axes[1, 0])
        sns.violinplot(data=paid_df, y=column_name, color='#fb9a99', ax=axes[1, 1])
        sns.violinplot(data=charged_default_df, y=column_name, color='#fb9a99', ax=axes[1, 2])
        sns.violinplot(data=risky_df, y=column_name, color='#fb9a99', ax=axes[1, 3])

        # Adding horizontal mean lines
        axes[1, 0].axhline(df[column_name].mean(), color='red', linestyle='dashed', linewidth=1.5, label='Mean')
        axes[1, 1].axhline(paid_df[column_name].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
        axes[1, 2].axhline(charged_default_df[column_name].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')        
        axes[1, 3].axhline(risky_df[column_name].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')

        pyplot.suptitle(column_name, fontsize='xx-large')
        pyplot.tight_layout()

        return pyplot.show()
