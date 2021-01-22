import json

# for comparing lists
import collections

# to handle datasets
import pandas as pd
import numpy as np

# to display all the columns of the dataframe
pd.pandas.set_option("display.max_columns", None)

from .plot.plot import Plot
from .commons import infer
from .commons import assure

class Analyzer():
    """Produce meta-data and plot features

    This class has the objective of producing meta-data about the dataset
    that should prove useful in selecting and engineering features before
    feeding it to the model.
    """
    def __init__(self, data, target, discrete_threshold=50, unique_threshold=0.9, rare_threshold=0.01,
                 outlier_threshold=3, skewness_threshold=0.5):
        """
        data: Pandas DataFrame
            Training Dataset with features and target

        target: string
            Target variable name

        discrete_threshold: int
            Variables that have number of unique values less than or equal to this value 
            will be considered discrete numerical variables that are not unique will be 
            considered continuous

        unique_threshold: float
            Variables that have unique value ratio greater than or equal to this value 
            will be considered to have mostly unique values, categorical variables with
            cardinality ratio higher than this will not be plotted in PreSci.plot_features()

        rare_threshold: float
            Values represented in a ratio lesser than or equal to this value for a variable 
            will be considered rare this will only be calculated for non-unique variables

        outlier_threshold: float
            Values with z-score greater or equal to this value will be considered outliers

            In other words, values that are more than or equal to "this value" times the 
            standard deviation for the variable, are considered outliers

            This has a standard value of 3, according to the statistical rule of thumb
            this will only be calculated for continuous variables

        skewness_threshold: float
            A floating point number to indicate the maximum skewness value for which
            the variable will be considered to have a normal distribution

            This is useful for normalizing distributions

            The default value is set to 0.5 according to the statistical convention that 
            states that it is not considered a normal distribution

            0~0.5 = normal distribution
            0.5~1 = slightly skewed distribution
            1~1.5 = skewed distribution
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")
        assure.type_equals([str], target, "target")
        assure.type_equals([int], discrete_threshold, "discrete_threshold")
        assure.type_equals([float, int], unique_threshold, "unique_threshold")
        assure.type_equals([float], rare_threshold, "rare_threshold")
        assure.type_equals([float, int], outlier_threshold, "outlier_threshold")
        assure.type_equals([float, int], skewness_threshold, "skewness_threshold")
        
        self.data = data
        self.target = target
        self.discrete_threshold = discrete_threshold
        self.unique_threshold = unique_threshold
        self.rare_threshold = rare_threshold
        self.skewness_threshold = skewness_threshold
        self.outlier_threshold = outlier_threshold

        self.meta_data = {
            "features": {}, 
            "target": {}, 
            "dataset": {}, 
            "categorical_features": [],
            "unique_features": [],
            "numerical_features": [],
            "discrete_features": [],
            "continuous_features": [],
            "skewed_features": [],
            "features_with_missing": [],
            "features_with_outliers": [],
            "all_features": [],
            "all_variables": [],
        }

        self.plot = Plot()

        self.__analyze_data()

    def get_meta_data(self):
        return self.meta_data

    def print_meta_data(self):
        print(json.dumps(self.meta_data, sort_keys=True, indent=2))

    def print_correlation(self):
        copy = self.data.copy()
        for var, info in self.meta_data["features"].items():
            if "categorical" in info:
                copy.loc[:,var] = copy.loc[:,var].astype("category").cat.codes
        if "categorical" in self.meta_data["target"]:
            copy.loc[:,self.target] = copy.loc[:,var].astype("category").cat.codes
        correlation_table = copy.corr(method="pearson")
        print(correlation_table)

    def plot_features(self):
        for var, info in self.meta_data["features"].items():
            if "continuous" in self.meta_data["target"]:
                if "discrete" in info:
                    self.plot.discrete_feature_continuous_target(self.data, var, self.target)
                if "continuous" in info:
                    self.plot.continuous_feature_continuous_target(self.data, var, self.target)
                    self.plot.continuous_distribution(self.data, var)
                if "categorical" in info and info["unique"] < self.unique_threshold:
                    self.plot.categorical_feature_continuous_target(self.data, var, self.target)
                if "missing_values" in info:
                    self.plot.na_continuous_target(self.data, var, self.target)
            else:
                if "discrete" in info:
                    self.plot.discrete_feature_non_continuous_target(self.data, var, self.target)
                if "continuous" in info:
                    self.plot.continuous_feature_non_continuous_target(self.data, var, self.target)
                    self.plot.continuous_distribution(self.data, var)
                if "categorical" in info and info["unique"] < self.unique_threshold:
                    self.plot.categorical_feature_non_continuous_target(self.data, var, self.target)
                if "missing_values" in info:
                    self.plot.na_non_continuous_target(self.data, var, self.target)

    def __analyze_data(self):
        numerical_vars = self.__list_numerical()
        categorical_vars = self.__list_categorical()
        
        self.__set_var_type(numerical_vars, "numerical")
        self.__set_var_type(categorical_vars, "categorical")

        self.__set_cardinality()
        self.__set_unique()
        self.__set_frequent_labels()        

        self.__set_discrete()
        self.__set_discrete_distribution()
        
        self.__set_continuous()
        self.__set_skewness()
        self.__set_outliers()

        self.__set_central_tendency()

        self.__set_target()
        self.__set_features()
        self.__set_variables()

        self.__set_na()

        self.__set_dataset()

    def __list_numerical(self):
        num_vars = [var for var in self.data.columns if self.data.loc[:,var].dtypes != "O"]
        for var in num_vars:
            if var != self.target:
                self.meta_data["numerical_features"].append(var) 
        return num_vars

    def __list_categorical(self):
        cat_vars = [var for var in self.data.columns if self.data.loc[:,var].dtypes == "O"]
        for var in cat_vars:
            if var != self.target:
                self.meta_data["categorical_features"].append(var)
        return cat_vars

    def __set_var_type(self, vars, category):
        for var in vars:
            self.meta_data["features"][var] = {category: True}

    def __set_cardinality(self):
        for var, info in self.meta_data["features"].items():
            cardinality = self.data.loc[:,var].nunique()
            info["cardinality"] = cardinality
            if cardinality == 2:
                info["binary"] = True

    def __set_unique(self):
        """Infer if categorical variable is unique according to threshold"""
        for var in self.meta_data["categorical_features"]:
            variable_isnull = self.data.loc[:,var].isnull()
            variable_filled_size = len(variable_isnull[variable_isnull == False].index)
            cardinality = self.meta_data["features"][var]["cardinality"]
            unique_ratio = cardinality / variable_filled_size
            self.meta_data["features"][var]["unique"] = unique_ratio
            if unique_ratio >= self.unique_threshold:
                self.meta_data["unique_features"].append(var)

    def __set_frequent_labels(self):
        for var in self.meta_data["categorical_features"]:
            frequent_labels = infer.frequent_labels(self.data.loc[:,var].copy(), self.rare_threshold)
            if not frequent_labels.empty:
                self.meta_data["features"][var]["frequent_labels"] = {}
                for frequent_label, occurrence in frequent_labels.to_dict().items():
                    self.meta_data["features"][var]["frequent_labels"][frequent_label] = occurrence

    def __set_discrete(self):
       for var, info in self.meta_data["features"].items():
            if "numerical" in info and info["cardinality"] <= self.discrete_threshold:
                info["discrete"] = True
                if var != self.target:
                    self.meta_data["discrete_features"].append(var)

    def __set_discrete_distribution(self):
        copy = self.data.copy()
        for var in self.meta_data["discrete_features"]:
            distribution = copy.loc[:,var].value_counts(normalize=True)
            self.meta_data["features"][var]["distribution"] = {}
            for value, occurrence in distribution.to_dict().items():
                self.meta_data["features"][var]["distribution"][value] = occurrence

    def __set_continuous(self):
        for var, info in self.meta_data["features"].items():
            if "numerical" in info and "discrete" not in info:
                info["continuous"] = True
                if var != self.target:
                    self.meta_data["continuous_features"].append(var)

    def __set_skewness(self):
        for var, info in self.meta_data["features"].items():
            if "continuous" in info:
                skewness = self.data.loc[:,var].skew()
                info["skewness"] = skewness
                if skewness > self.skewness_threshold and var != self.target:
                    self.meta_data["skewed_features"].append(var)

    def __set_outliers(self):
        """Discerns outliers in continuous variables

        Writes to meta-data all lines of variable that have the absolute zscore higher than the outlier_threshold
        """
        for var, info in self.meta_data["features"].items():
            if "continuous" in info:
                var_series = self.data.loc[:,var].copy()
                outliers = infer.outliers(var_series, self.outlier_threshold)
                if not outliers.empty:
                    info["outliers"] = {}
                    info["outliers"]["min"] = outliers.min()
                    info["outliers"]["max"] = outliers.max()
                    info["outliers"]["mean"] = outliers.mean()
                    info["outliers"]["median"] = outliers.median()
                    info["outliers"]["ratio"] = len(outliers.index) / len(var_series.index)
                    if var != self.target:
                        self.meta_data["features_with_outliers"].append(var)

    def __set_central_tendency(self):
        for var, info in self.meta_data["features"].items():
            if "continuous" in info:
                info["mean"] = self.data.loc[:,var].mean(skipna=True)
                info["median"] = self.data.loc[:,var].median(skipna=True)
                info["mode"] = self.data.loc[:,var].mode(dropna=True).tolist()
            else:
                if "categorical" in info and info["unique"] >= self.unique_threshold:
                    continue 
                else:
                    info["mode"] = self.data.loc[:,var].mode(dropna=True).tolist()

    def __set_target(self):
        target = self.meta_data["features"].pop(self.target)
        self.meta_data["target"] = target
        self.meta_data["target"]["name"] = self.target

    def __set_features(self):
        """ Makes list of features

        This method makes a feature list in the order they were inputed
        in the training dataframe.

        It's important to keep all columns in the same order because some 
        functionalities are based on this order.
        """
        for feature in self.data.drop(self.target, axis=1).columns:
            self.meta_data["all_features"].append(feature)

    def __set_variables(self):
        """ Makes list of variables

        This method makes a variable list in the order they were inputed
        in the training dataframe.

        It's important to keep all columns in the same order because some 
        functionalities are based on this order.
        """
        for variable in self.data.columns:
            self.meta_data["all_variables"].append(variable)

    def __set_na(self):
        vars_with_na = [var for var in self.data.columns if self.data.loc[:,var].isnull().sum() > 0]
        for var in vars_with_na:
            if var != self.target:
                self.meta_data["features_with_missing"].append(var)

        missing_ratios = self.data.loc[:,vars_with_na].isnull().mean().to_dict()
        na_vars_info = {}
        for var in vars_with_na:
            if "continuous" in self.meta_data["target"]:
                na_vars_info[var] = self.__analyse_na_numerical_target(self.data, var, missing_ratios[var])
            else:
                na_vars_info[var] = self.__analyse_na_categorical_target(self.data, var, missing_ratios[var])

        for var, na_info in na_vars_info.items():
            self.meta_data["features"][var]["missing_values"] = na_info

    def __analyse_na_numerical_target(self, df, var, missing_ratio):
        join = df.loc[:,[var, self.target]]
        join.loc[:,var] = join.loc[:,var].isnull().replace({True:"missing", False:"filled"})
        unique_variable_values = join.loc[:,var].unique()
        unique_variable_values.sort()

        target_mean_per_variable_value = {}
        for variable_value in unique_variable_values:
            rows_variable_value = join[join.loc[:,var] == variable_value]
            target_only = rows_variable_value[self.target]
            mean = target_only.mean()
            target_mean_per_variable_value[variable_value] = round(mean,2)

        na_info = {"missing_ratio": missing_ratio, "target_mean": target_mean_per_variable_value}
        return na_info

    def __analyse_na_categorical_target(self, df, var, missing_ratio):
        join = df.loc[:,[var, self.target]]
        join.loc[:,var] = join.loc[:,var].isnull().replace({0: "filled", 1: "missing"})
        unique_variable_values = join.loc[:,var].dropna().unique()
        unique_variable_values.sort()
        unique_target_values = join.loc[:,self.target].dropna().unique()
        unique_target_values.sort()

        relation = {}
        for variable_value in unique_variable_values:
            relation[variable_value] = {}
            for target_value in unique_target_values:
                variable_value_size_for_target_value = len(join[(join.loc[:,self.target] == target_value) & (join.loc[:,var] == variable_value)].index)
                variable_value_size = len(join[join.loc[:,var] == variable_value].index)
                ratio = variable_value_size_for_target_value / variable_value_size
                relation[variable_value][str(target_value)] = ratio

        na_info = {"missing_ratio": missing_ratio, "ratio_per_target_value": relation}
        return na_info

    def __set_dataset(self):
        self.meta_data["dataset"]["shape"] = self.data.shape
