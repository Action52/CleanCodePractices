# Predict Customer Churn
* * *
- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity.  

## Project Description
This project implements a library that executes analysis on a bank dataset 
(provided in the repo). The library creates a model, stores it, and saves analysis
plots in the folder.

## Files and data description
Important files and folders are:

- _data/bank_data.csv_: The provided dataset to experiment with.
- _churn_detector_library/utils.py_: This is the actual library implementing all the functions from the notebook.
- _images_: This folder stores the _eda_ and _results_ plots.
- _logs_: This folder stores the log data.
- _models_: This folder stores the models to easy load in future instances.
- _churn_notebook.ipynb_: The original notebook to create the library.
- _churn_script_logging_and_tests.py_: The testing script to use with pytest.
- _conftest.py_: This file sets up pytest to work with the testing file.
- _config.yaml_: This file stores important variables that can be modified without updating the scripts. 

## Running Files
- Download the repo folder, and on a terminal navigate to the repo root.
- First create an empty environment and install the libraries. Let's use conda as an example.  
```
conda create --name churn-detector python=3.8
pip install -r requirements_py3.8.txt
```

- Now you can run the testing files:
```
pytest churn_script_logging_and_tests.py
```

- The previous command will execute a test run and generate the appropriate plots.




