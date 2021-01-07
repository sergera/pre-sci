# to handle datasets
import pandas as pd
import numpy as np

# for comparing lists
import collections

# for finding outliers
from scipy.stats import zscore

# to display all the columns of the dataframe
pd.pandas.set_option("display.max_columns", None)

class Analyzer():
    """Produce meta-data

    This class has the objective of producing meta-data about the dataset
    that should prove useful in selecting and engineering features before
    feeding it to the model.
    """
    def __init__(self, data, target, discrete_threshold, unique_threshold, rare_threshold,
                 outlier_threshold, skewness_threshold):
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
            "all_categorical": [],
            "all_unique": [],
            "all_numerical": [],
            "all_discrete": [],
            "all_continuous": [],
            "all_skewed": [],
            "all_with_missing": [],
            "all_with_outliers": [],
            "all_features": [],
            "all_variables": [],
        }

        self.__analyze_data()

    def __analyze_data(self):
        categorical_vars = self.__list_categorical()
        numerical_vars = self.__list_numerical()
        
        self.__set_var_type(categorical_vars, "categorical")
        self.__set_var_type(numerical_vars, "numerical")
        self.__set_cardinality()
        
        self.__set_discrete()
        self.__set_continuous()
        
        self.__set_central_tendency()
        self.__set_skewness()
        self.__set_outliers()


        self.__set_unique()

        self.__set_frequent_labels()

        self.__set_target()
        self.__set_features()
        self.__set_variables()

        self.__set_na()

        self.__set_dataset()

    def get_meta_data(self):
        return self.meta_data

    def __list_categorical(self):
        cat_vars = [var for var in self.data.columns if self.data.loc[:,var].dtypes == "O"]
        self.meta_data["all_categorical"] = cat_vars
        return cat_vars

    def __list_numerical(self):
        num_vars = [var for var in self.data.columns if self.data.loc[:,var].dtypes != "O"]
        self.meta_data["all_numerical"] = num_vars
        return num_vars

    def __set_var_type(self, vars, category):
        for var in vars:
            self.meta_data["features"][var] = {category: True}

    def __set_cardinality(self):
        for var, info in self.meta_data["features"].items():
            cardinality = self.data.loc[:,var].nunique()
            info["cardinality"] = cardinality
            if cardinality == 2:
                info["binary"] = True

    def __set_discrete(self):
       for var, info in self.meta_data["features"].items():
            if "numerical" in info and info["cardinality"] <= self.discrete_threshold:
                info["discrete"] = True
                self.meta_data["all_discrete"].append(var)

    def __set_continuous(self):
        for var, info in self.meta_data["features"].items():
            if "numerical" in info and "discrete" not in info:
                info["continuous"] = True
                self.meta_data["all_continuous"].append(var)

    def __set_central_tendency(self):
        for var, info in self.meta_data["features"].items():
            if "continuous" in info:
                info["mean"] = self.data.loc[:,var].mean(skipna=True)
                info["median"] = self.data.loc[:,var].median(skipna=True)
                info["mode"] = self.data.loc[:,var].mode(dropna=True).tolist()
            else:
                info["mode"] = self.data.loc[:,var].mode(dropna=True).tolist()

    def __set_skewness(self):
        for var, info in self.meta_data["features"].items():
            if "continuous" in info:
                skewness = self.data.loc[:,var].skew()
                info["skewness"] = skewness
                if skewness > self.skewness_threshold:
                    self.meta_data["all_skewed"].append(var)

    def __set_outliers(self):
        """Discerns outliers from continuous variables

        Writes to meta-data all lines of variable that have the absolute zscore higher than the outlier_threshold
        """
        for var, info in self.meta_data["features"].items():
            if "continuous" in info:
                var_series = self.data.loc[:,var].dropna()
                outliers = var_series[(np.abs(zscore(var_series)) >= self.outlier_threshold)]
                info["outliers"] = outliers.to_dict()
                self.meta_data["all_with_outliers"].append(var)

    def __set_unique(self):
        """Infer if categorical variable is unique according to threshold"""
        for var in self.meta_data["features"]:
            variable_isnull = self.data.loc[:,var].isnull()
            variable_filled_size = len(variable_isnull[variable_isnull == False].index)
            cardinality = self.meta_data["features"][var]["cardinality"]
            unique_ratio = cardinality / variable_filled_size
            self.meta_data["features"][var]["unique"] = unique_ratio
            if unique_ratio >= self.unique_threshold:
                self.meta_data["all_unique"].append(var)

    def __set_frequent_labels(self):
        frequent_labels_by_var = {}
        for var in self.meta_data["all_categorical"]:
            frequent_labels = self.__analyse_frequent_labels(var)
            if not frequent_labels.empty:
                self.meta_data["features"][var]["frequent_labels"] = {}
                for frequent_label, occurrence in frequent_labels.to_dict().items():
                    self.meta_data["features"][var]["frequent_labels"][frequent_label] = occurrence

    def __analyse_frequent_labels(self, var):
        category_distribution = self.data.copy().groupby(var)[self.target].count() / len(self.data.loc[:,var].dropna().index)
        frequent_labels = category_distribution[category_distribution > self.rare_threshold]
        return frequent_labels

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
        self.meta_data["all_with_missing"] = vars_with_na
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
            rows_variable_value = join.loc[:,join.loc[:,var] == variable_value]
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
