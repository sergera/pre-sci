import numpy as np

# for finding outliers
from scipy.stats import zscore

def frequent_labels(series, rare_threshold):
    """
    series: Pandas Series
        Pandas Series where the frequent labels inference will be made

    rare_threshold: float
        Labels with frequency greater than this value will be considered frequent

    returns: Pandas Series
        Pandas Series with frequent labels and their frequency
    """
    category_distribution = series.value_counts(normalize=True)
    frequent_labels = category_distribution[category_distribution > rare_threshold]
    return frequent_labels

def outliers(series, outlier_threshold):
    """
    series: Pandas Series
        Pandas Series to select outlier rows from

    outliers_threshold: float
        Values with z-score greater or equal to this value will be considered outliers

        In other words, values that are more than or equal to "this value" times the 
        standard deviation for the variable, are considered outliers

    returns: Pandas Series
        Pandas Series with outlier rows
    """
    series = series.dropna()
    outliers = series[(np.abs(zscore(series)) >= outlier_threshold)]
    return outliers