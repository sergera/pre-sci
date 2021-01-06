# for printing meta_data dict
import json

import pandas as pd
import numpy as np

from .analyzer.analyzer import Analyzer
from .plot.plot import Plot
from .transformer.transformer import Transformer

class PreSci():
    """Get to the part that really matters faster, the science!

    This package aims to help in the analysis and pre-processing 
    of data for predictive modeling.

    !!!IMPORTANT!!!
    It is made for use with continuous and binary targets only!
    Targets must be fed as numeric to PreSci!

    Import:
        from pre-sci.presci import PreSci

    After instantiating it:
        presci = PreSci(data, target_name)

    You can easily plot features of a given dataset with:
        presci.plot_features()

    You can print a meta data dictionary to the console with:
        presci.print_meta_data()

    You can print a correlation table with:
        presci.print_correlation()

    You can also infer on the data after the transformations with:
        presci.post_plot_features()
        presci.post_print_meta_data()
        presci.post_print_correlation()

    You can transform data for training with:
        presci.fit_transform()

    And for predicting with:
        presci.transform(data)

    On feature engineering, this package replaces rare labels on categorical data, 
    encodes categorical (and discrete, if requested) data, replaces missing data using MICE, 
    normalizes the distribution of data with high skewness, embeds data using NNs (if requested), 
    and scales data, in order.

    !!!IMPORTANT!!!
    If you should choose to ordinally encode your categorical (or discrete) data, 
    and you have some criteria to do so, I strongly encourage you to encode according 
    to your criteria before feeding the data into PreSci!

    PreSci has an ordinal encoder and an auto-encoder, the ordinal encoder uses no
    criteria, the auto-encoder works as follows:

    If categorical variables are fed to PreSci in string form, and not parametrized 
    as variables to be encoded in some form, PreSci will auto-encode them in relation 
    to the target value in case the target is continuous or binary, if target is neither,
    PreSci will ordinally encode variable with no criteria!!!

    You can also especically ask for the auto-encoder to be used using the "auto_encode"
    parameter (see constructor docs below).

    With continuous targets, variable will be labeled according to target mean, 
    and with binary targets the variable will be labeled according to incidence of
    highest target value, which is 1 (labels will be ordered by the ratio of target 
    value for that label).

    Any categorical label unknown to the auto-encoder will be labeled as 0!
    Non-categorical unknown values will be left as NaN for the MICE model to predict them.

    Auto Encoder Example: 
        Continuous Target:
            mean of target value for label a is 50
            mean of target value for label b is 80
            nean of target value for label c is 100
            encoded values are {a:0, b:1, c:2}

        Binary Target:
            10% of label a has target == 1
            30% of label b has target == 1
            60% of label c target == 1
            encoded values are {a:0, b:1, c:2}
    """

    def __init__ (self, data, target, discrete_threshold=20, unique_threshold=0.9, 
        rare_threshold=0.01, outlier_threshold=3, skewness_threshold=0.5, 
        to_onehot_encode=[], to_ordinal_encode=[], to_embed=[], to_auto_encode=[], 
        custom_encoders={}, dont_scale=[], dont_normalize=[], seed=0, callback=None, 
        remove_outliers=False, test_size=0.1):
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

        remove_outliers: bool
            This will decide whether the outliers found with the outlier threshold will
            have their rows removed from the training data

        skewness_threshold: float
            A floating point number to indicate the maximum skewness value for which
            the variable will be considered to have a normal distribution

            This is useful for normalizing distributions

            The default value is set to 0.5 according to the statistical convention that 
            states that it is not considered a normal distribution

            0~0.5 = normal distribution
            0.5~1 = slightly skewed distribution
            1~1.5 = skewed distribution

        to_onehot_encode: [string]
            List of variable names to be onehot encoded

        to_ordinal_encode: [string]
            List of variable names to be ordinal encoded

        to_embed: [string]
            List of variables to be embedded using with a neural network

        to_auto_encode: [string]
            List of variables to be auto_encoded according to target value, 
            variables should be categorical or discrete, and should not include target,
            note that auto_encode only works with continuous or binary targets

        custom_encoders: {variable: {label: encoding}}
            A dictionary of variables to be encoded by the user, each key is a variable name,
            and each value is a dictionary that has current labels as keys and encoded values
            as values.
            This can be used for categorical or discrete variables.
            Put ALL KNOWN VALUES in the encoder, any forgotten values will be treated as unknown.

        dont_scale: [string]
            List of variable names to not be scaled by transformer, this will only work for 
            the final scalling, if the variable is to be embedded it will be scaled for NN 
            processing

        dont_normalize: [string]
            List of variable names to not be normalized (log()) by transformer, even if the
            variable distribution is skewed

        seed: int
            Random state to be used in all probabilistic functionalities to ensure
            reproducibility of the trained model

        callback: func
            This is a custom transformation that will be applied to the dataset before 
            the training and predicting transformations, it should only be applied to features,
            and should have a parameter, that will be the dataset, and return this parameter
            in the end.

            This function will be called before everything (even analysis!), this means that 
            the plotted features will have the characteristics given by this function. So 
            don't do here, anything you wouldn't want to see in a graph. 
            
            It should contain feature selection, and preliminary transformations, any more 
            advanced transformation (like encodings) should be done with the encoding auto 
            parameters like "to_onehot_encode", and "to_ordinal_encode" and "to_auto_encode",
            or with "custom_encoders" for increased control (read above).
            
            Callback Example:

            def custom_transform(data):
                data.drop(["column_a", "column_b"], inplace=True, axis=1)
                data.loc[:,"column_c"] = data.loc[:,"column_c"].apply(lambda x: x + 1)
                return data
        """
        self.target = target
        self.discrete_threshold = discrete_threshold
        self.unique_threshold = unique_threshold
        self.rare_threshold = rare_threshold
        self.outlier_threshold = outlier_threshold
        self.remove_outliers = remove_outliers
        self.skewness_threshold = skewness_threshold
        self.to_onehot_encode = to_onehot_encode
        self.to_ordinal_encode = to_ordinal_encode
        self.to_embed = to_embed
        self.to_auto_encode = to_auto_encode
        self.custom_encoders = custom_encoders
        self.dont_scale = dont_scale
        self.dont_normalize = dont_normalize
        self.seed = seed
        self.test_size = test_size
        self.callback = callback

        def copy_data(func):
            def wrapper(data):
                copied_data = data.copy()
                return func(copied_data)
            return wrapper

        if self.callback:
            # if there is a transform callback
            # make sure data is copied
            self.callback = copy_data(self.callback)
            # analyzer must recieve the transformed data
            self.data = self.callback(data)
        else:
            self.data = data

        self.analysis = Analyzer(
            self.data, 
            target, 
            discrete_threshold, 
            unique_threshold, 
            rare_threshold, 
            outlier_threshold, 
            skewness_threshold,
        )

        self.meta_data = self.analysis.get_meta_data()

        self.transformer = Transformer(
            skewness_threshold=skewness_threshold,
            seed=seed,
            test_size=self.test_size,
        )

        # replacing rare labels from training data here 
        # so that the user can see them when plotting
        # rare labels from predicting data will be dealt with in the "transform" method
        self.original_data = self.replace_rare_labels(self.data.copy())
        self.plot = Plot()

        # this will be a new analyzer so the user can infer on the
        # data post transforms
        self.post_analysis = None
        self.post_meta_data = None
        self.post_data = None

    def transform_fit(self):
        """Transform for training

            - Replaces rare categorical labels with "Rare"
            - Fits onehot encoder models
            - Fits ordinal encoder models
            - Encodes categorical variables
            - Fits mice model with training data for NA value treatment
            - Replaces missing data
            - Fits categorical embedding NNs
            - Embeds variables
            - Normalizes distributions by logging continuous variables with high skewness
            - Fits scaler models for every variable
            - Scales variable values
            - Transforms for training (with split)
        """
        data = self.data

        self.set_original_names(data)
        data = self.replace_rare_labels(data)

        if self.remove_outliers:
            data = self.remove_outlier_rows(data)

        self.fit_encoders(data)
        data = self.encode(data)

        # sklearn's IterativeImputer (MICE) depends on the order of columns
        data = data.sort_index(axis=1)
        features_only = data.drop(self.target, axis=1)
        self.fit_mice(features_only)
        data.update(self.replace_missing(features_only))

        self.fit_embedders(data)
        data = self.embed(data)

        data = self.normalize(data)

        self.fit_scalers(data)
        data = self.scale(data)

        self.save_post_analysis(data)

        X_train, X_test, y_train, y_test = self.split(data)
        return X_train, X_test, y_train, y_test

    def transform(self, data):
        """Transform for predicting

        data: Pandas DataFrame
            Feature dataset to be fed to the model for prediction

            - Replaces rare categorical labels with "Rare"
            - Encodes categorical variables
            - Replaces missing data
            - Normalizes distributions by logging continuous variables with high skewness
            - Embeds variables
            - Scales variable values
        """
        if self.callback:
            data = self.callback(data)

        data = data.loc[:,self.meta_data["all_features"]]
        data = self.replace_rare_labels(data)
        data = self.encode(data)

        # sklearn's IterativeImputer (MICE) depends on the order of columns
        data = data.sort_index(axis=1)
        data.update(self.replace_missing(data))

        data = self.embed(data)
        data = self.normalize(data)   
        data = self.scale(data)
        return data

    def print_meta_data(self):
        self.__print_meta_data(self.meta_data)

    def plot_features(self):
        self.__plot_features(self.original_data, self.meta_data)
    
    def print_correlation(self):
        data = self.data.copy()
        for var, info in self.meta_data["features"].items():
            if "categorical" in info:
                data.loc[:,var] = data.loc[:,var].astype("category").cat.codes
        if "categorical" in self.meta_data["target"]:
            data.loc[:,self.target] = data.loc[:,var].astype("category").cat.codes
        correlation_table = data.corr(method="pearson")
        print(correlation_table)

    def post_print_meta_data(self):
        try:
            self.__print_meta_data(self.post_meta_data)
        except Exception as UntimedAccessException:
            raise Exception(
                """Cannot print meta_data post transformations!
                Data has not been transformet yet!
                Call 'presci.fit_transform()'!""")

    def post_plot_features(self):
        try:
            self.__plot_features(self.post_data ,self.post_meta_data)
        except Exception as UntimedAccessException:
            raise Exception(
                """Cannot plot features post transformations!
                Data has not been transformet yet!
                Call 'presci.fit_transform()'!""")
            

    def post_print_correlation(self):
        try:
            correlation_table = self.post_data.corr(method="pearson")
            print(correlation_table)    
        except Exception as UntimedAccessException:
            raise Exception(
                """Cannot print correlation post transformations!
                Data has not been transformet yet!
                Call 'presci.fit_transform()'!""")

    def __print_meta_data(self, meta_data):
        print(json.dumps(meta_data, sort_keys=True, indent=2))

    def __plot_features(self, data, meta_data):
        for var, info in meta_data["features"].items():
            if "continuous" in meta_data["target"]:
                if "discrete" in info:
                    self.plot.discrete_feature_continuous_target(data, var, self.target)
                if "continuous" in info:
                    self.plot.continuous_feature_continuous_target(data, var, self.target)
                    self.plot.continuous_distribution(data, var)
                if "categorical" in info and info["unique"] < self.unique_threshold:
                    self.plot.categorical_feature_continuous_target(data, var, self.target)
                if "missing_values" in info:
                    self.plot.na_continuous_target(data, var, self.target)
            else:
                if "discrete" in info:
                    self.plot.discrete_feature_non_continuous_target(data, var, self.target)
                if "continuous" in info:
                    self.plot.continuous_feature_non_continuous_target(data, var, self.target)
                    self.plot.continuous_distribution(data, var)
                if "categorical" in info and info["unique"] < self.unique_threshold:
                    self.plot.categorical_feature_non_continuous_target(data, var, self.target)
                if "missing_values" in info:
                    self.plot.na_non_continuous_target(data, var, self.target)

    def set_original_names(self, data):
        self.transformer.set_original_names(data)

    def replace_rare_labels(self, data):
        for var_name, info in self.meta_data["features"].items():
            if "frequent_labels" in info:
                data = self.transformer.replace_rare_labels(data, var_name, info["frequent_labels"])
        return data

    def remove_outlier_rows(self, data):
        for var_name, info in self.meta_data["features"].items():
            if "outliers" in info:
                data = self.transformer.remove_outlier_rows(data, info["outliers"])
        return data

    def fit_encoders(self, data):
        # check if there are any variables set for different encoders, which is not allowed
        # with the exception of ordinal-embed, and custom-embed, since variables must be encoded before embeded
        intersections = {}
        intersections["onehot_ordinal"] = set.intersection(set(self.to_onehot_encode), set(self.to_ordinal_encode))
        intersections["onehot_custom"] = set.intersection(set(self.to_onehot_encode), set(self.custom_encoders))
        intersections["onehot_embed"] = set.intersection(set(self.to_embed), set(self.to_onehot_encode))
        intersections["ordinal_embed"] = set.intersection(set(self.to_embed), set(self.to_ordinal_encode))
        intersections["ordinal_custom"] = set.intersection(set(self.to_ordinal_encode), set(self.custom_encoders))
        intersections["custom_embed"] = set.intersection(set(self.to_embed), set(self.custom_encoders))
        for intersection_name, intersection_value in intersections.items():
            if intersection_value:
                raise Exception(
                """Same variable was set to encode in more than one encoder!""")

        # check if auto-encoder is possible, if not use ordinal-encoding
        auto_encoder_possible = False
        if "continuous" in self.meta_data["target"]:
            fit_auto_encoder = self.transformer.fit_auto_encoder_continuous_target
            auto_encoder_possible = True
        elif "binary" in self.meta_data["target"]:
            fit_auto_encoder = self.transformer.fit_auto_encoder_binary_target
            auto_encoder_possible = True

        for var_name, info in self.meta_data["features"].items():
            # set categorical vars that were not set to encode to be auto-encoded
            # if target is not binary nor continuous set them to be ordinal encoded
            # if categorical var is set to be embedded it will also be encoded before
            var_not_set_to_encode = var_name not in self.to_onehot_encode and var_name not in self.to_ordinal_encode and var_name not in self.custom_encoders
            if "categorical" in info and var_not_set_to_encode:
                if var_name not in self.to_auto_encode:
                    if auto_encoder_possible:
                        self.to_auto_encode.append(var_name)
                    elif not auto_encoder_possible:
                        self.to_ordinal_encode.append(var_name)

        fit_auto_encoder(data, self.to_auto_encode, self.target)
        self.transformer.fit_onehot_encoder_model(data, self.to_onehot_encode)
        self.transformer.fit_ordinal_encoder_model(data, self.to_ordinal_encode)
        self.transformer.set_custom_encoders(self.custom_encoders)

    def encode(self, data):
        data = self.transformer.encode_all(data, self.meta_data["features"])
        return data

    def fit_mice(self, features):
        self.transformer.fit_mice_model(features)

    def replace_missing(self, data):
        data = self.transformer.replace_na(data)
        return data

    def fit_embedders(self, data):
        self.transformer.fit_embedder_model(data, self.to_embed, self.target)

    def embed(self, data):
        data = self.transformer.embed_categorical(data)
        return data

    def normalize(self, data):
        data = self.transformer.normalize_distribution(data, self.target, self.dont_normalize, self.meta_data)
        return data

    def fit_scalers(self, data):
        self.transformer.fit_scaler_models(data, self.dont_scale, self.meta_data["all_continuous"])

    def scale(self, data):
        data = self.transformer.scale(data, self.dont_scale)
        return data

    def save_post_analysis(self, data):
        # save analysis post transformation
        self.post_analysis = Analyzer(
            data, 
            self.target, 
            self.discrete_threshold, 
            self.unique_threshold, 
            self.rare_threshold, 
            self.outlier_threshold, 
            self.skewness_threshold,
        )
        self.post_meta_data = self.post_analysis.get_meta_data()
        self.post_data = data

    def split(self, data):
        X_train, X_test, y_train, y_test = self.transformer.split(data, self.target)
        return X_train, X_test, y_train, y_test