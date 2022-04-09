from logging import critical
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2, pointbiserialr 

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def q95(x):
    return x.quantile(0.95)

def q99(x):
    return x.quantile(0.99)

def groupbySummary(df, target, col):
  """
  Takes df and groups by target categorical variable and displays the following summary statistics for the col continuous variable: max, min, median, mean, std, 25th quantile, 75th quantile, 95th quantile, and the 99th quantile. 
  """
  return df.groupby(target)[col].agg(Max=np.max, Min=np.min, Mean=np.mean, Std=np.std, Q25=q25, Q75=q75, Q95=q95, Q99=q99)

def groupbyCounts(df, target, col):
  """
  Takes df and groups by targer categorical variable and displays normalized value counts in the form of a percent on the occurences between the target and col variable possibilities.
  """
  return df.groupby(col)[target].value_counts(normalize=True).mul(100).rename('percent').reset_index()

def buildContingencyTable(df, target, col):
  """
  Builds a contingency table and returns values normalized based on the col value. This will help visualize quickly if the target variable occurs more frequently in certain observations of col.
  """
  return pd.crosstab(df[target], df[col], normalize='columns')

def computeChiSquareStatistics(contTable):
  """
  Computes Chi Square statistics; returns the test statistic, the critical value, and the p-value for the test
  """
  test_stat, p_val, dof, expected = chi2_contingency(contTable)
  crit_val = chi2.ppf(1-0.5, df=dof)
  return test_stat, crit_val, p_val

def computePointBiserialR(df, target, col):
  """
  Computes the Point Biserial R correlation for a categorical target and a continuous col.
  """
  return pointbiserialr(df[target], df[col])
 
