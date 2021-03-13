# CA04-Ensemble_Model
Build a ensemble models, optimize model, and measure model performance
# Description
The dataset is obtained from the Census Bureau and represents salaries of people along with seven demographic variables. The following is a description of our dataset:

•	Number of target classes: 2 ('>50K' and '<=50K') [ Labels: 1, 0 ]
•	Number of attributes (Columns): 7
•	Number of instances (Rows): 48,842
## Instructions
1. Download census_data.csv files to local computer
2. Open ipynb file
3. Click "Open in Colab" button or click the Colab link
4. Upload the file to goolge drive
5. Run code by lines or select runtime in the menu bar - click run all in the dropdown list
## Usage
import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier 

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

## Questions answered

Q.1 Write your observations about the Classifier’s behavior with respect to the
number of estimators

Q.2 Is there an optimal value of the estimator within the given range?
