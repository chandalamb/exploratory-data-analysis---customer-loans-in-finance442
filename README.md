# Exploratory Data Analysis - Customer Loans in Finance

## Description

This Ai Core project aims to conduct Exploratory Data Analysis (EDA) on customer loan payment data. The process begins with extracting the data from an AWS Relational Database and transforming it into a pandas dataframe and a CSV file for subsequent processing and analysis.

The data undergoes several transformations including imputation and removal of null values, optimisation of skewness, outlier removal, and identification of correlations. After this, thorough analysis and visualization are performed to gain insights into the current state of loans, potential losses and indicators of risk.

The primary objective of this project is to delve into the loans_payment database through EDA to uncover underlying patterns, address data anomalies, employ statistical techniques for understanding data distribution, utilize visualization methods to discern patterns and trends, and ultimately present the findings in a clear and concise manner.


## Installation instructions

- Download and clone the repository:
  * Copy the repository URL from GitHub
  * In your command line interface (CLI), navigate to the desired directory
  * Use the 'git clone' command followed by the copied HTTPS URL
  * Press 'Enter' to clone the repository.
- Ensure the presence of the 'environment.yaml' file, which contains all necessary package versions for running the code. Using conda in your CLI, execute the 'conda env create -f environment.yml' command to create the environment, optionally specifying a name with the --name flag.

## File structure of the project
* db_utils.py: Python script used to extract data from an AWS RDS using confidential .yaml credentials. It has already been executed, resulting in the creation of 'loan_payments.csv' included in the repository.
* datatransform.py: Python script defining the DataTransform() class for transforming dataframe formats. Imported as a module into 'EDA.ipynb' notebook.
* dataframeinfo.py: Python script defining the DataFrameInfo() class for retrieving information and insights from the dataframe. Imported as a module into 'EDA.ipynb' notebook.
* dataframetransform.py: Python script defining the DataFrameTransformation() class for conducting dataframe transformations. Imported as a module into 'EDA.ipynb' notebook.
* plotter.py: Python script defining the Plotter() class for providing visualizations on the dataframe. Imported as a module into 'EDA.ipynb' notebook.
* EDA.ipynb: Notebook where exploratory data analysis (EDA) and dataframe transformation processes are conducted. It should be executed and reviewed to understand the EDA process.
* analysis_and_visualisation.ipynb: Notebook containing analysis and visualizations of the transformed dataframe. It provides interactive insights and conclusions drawn from the data.

## License information

MIT License Copyright (c) 2024 Chanda Lamb

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “hangman milestones”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
