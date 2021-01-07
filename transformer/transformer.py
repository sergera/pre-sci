import numpy as np
# ignore divide by zero warning given by divisions avoided with np.where
np.seterr(divide = 'ignore')

import math

import operator

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy

from .embedder import Embedder

class Transformer:
    def __init__(self, test_size, seed, skewness_threshold=0.5):
        """
        test_size: float
            Ratio of dataset that will be split for testing during training

        skewness_threshold: float
            A floating point number to indicate the maximum skewness value for which
            the variable will be considered to have a normal distribution

            Variables with skewness over this value will have values replaced with their 
            natural logarithm (except if the value is 0), and will be scaled differently
        """
        self.test_size = test_size
        self.seed = seed
        self.skewness_threshold = skewness_threshold

        # multiple imputation by chained equations model
        self.mice = IterativeImputer(max_iter=100, random_state=self.seed, sample_posterior=True, verbose=1, min_value=0)

        # dict of onehot encoder models for features with low cardinality
        self.onehot_encoder_models = {}
        self.to_onehot_encode = []
        # dict of ordinal encoder models for each categorical feature to be embedded
        self.ordinal_encoder_models = {}
        self.to_ordinal_encode = []
        # dict of auto encoders made in relation to target value according to it's type
        self.auto_encoders = {}
        self.to_auto_encode =[]
        # dict of custom encodings to be used
        self.custom_encoders = {}
        self.to_custom_encode = []
        # dict of categorical embedder NN models for each categorical variable
        self.embedder_models = {}
        self.to_embed = []
        # dict of scaler models specifically used for scaling features being fed to NNs
        self.NN_scaler_models = {}

        # dict of scaler models for each variable 
        self.scaler_models = {}
        
        # dict of original feature names for all features that suffer redimensioning
        # key is original name, value is list with new column names
        self.original_names = {}

    def set_original_names(self, data):
        for column in data.columns:
            self.original_names[column] = []

    def replace_rare_labels(self, data, var_name, frequent_labels):
        all_labels = data[var_name].dropna().unique()
        for label in all_labels:
            if label not in frequent_labels:
                data.loc[:,var_name] = data[var_name].replace({label: "Rare"})    
        return data

    def remove_outlier_rows(self, data, outliers):
        for index in outliers:
            data = data.copy().drop(index ,axis="index")

        data.reset_index(inplace=True, drop=True)
        return data

    def fit_onehot_encoder_model(self, data, to_onehot):
        for var_name in to_onehot:
            self.onehot_encoder_models[var_name] = OneHotEncoder(handle_unknown="ignore")
            self.onehot_encoder_models[var_name].fit(data.loc[:,var_name].dropna().values.reshape(-1,1))
            self.to_onehot_encode.append(var_name)

    def fit_ordinal_encoder_model(self, data, to_ordinal):
        for var_name in to_ordinal:
            self.ordinal_encoder_models[var_name] = OrdinalEncoder(handle_unknown="ignore")
            self.ordinal_encoder_models[var_name].fit(data.loc[:,var_name].dropna().values.reshape(-1,1))
            self.to_ordinal_encode.append(var_name)

    def fit_auto_encoder_continuous_target(self, data, to_auto, target):
        for var_name in to_auto:
            ordered_labels = data.copy().groupby([var_name])[target].mean().sort_values().index
            ordinal_labels = {k: i for i, k in enumerate(ordered_labels, 0)}
            self.auto_encoders[var_name] = ordinal_labels
            self.to_auto_encode.append(var_name)

    def fit_auto_encoder_binary_target(self, data, to_auto, target):
        for var_name in to_auto:
            feature_labels = data[var_name].dropna().unique()
            feature_size = len(data[var_name].copy().dropna().index)
            max_target = data[target].unique().max()
            
            label_map = {}
            for label in feature_labels:
                label_size = len(data[(data.loc[:,var_name] == label)].index)
                occurences = len(data[(data.loc[:,var_name] == label) & (data.loc[:,target] == max_target)].index)
                label_map[label] = occurences / label_size

            # TO DO: decide whether to use soft or hard auto encodings for binary targets
            # soft = label_map, and hard = sorted_label_map
            sorted_labels = sorted(label_map.items(), key=lambda item: item[1], reverse=False)
            sorted_label_map = {k[0]: v for v, k in enumerate(sorted_labels, 0)}

            self.auto_encoders[var_name] = label_map
            self.to_auto_encode.append(var_name)

    def set_custom_encoders(self, custom_encoders):
        self.custom_encoders = custom_encoders
        for variable in custom_encoders:
            self.to_custom_encode.append(variable)

    def encode_all(self, data, features_info):
        for var_name in self.to_onehot_encode:
            data = self.__onehot_encode_existing_values(data, var_name)
        for var_name in self.to_ordinal_encode:
            data = self.__ordinal_encode_existing_values(data, var_name)

        def is_missing(str_or_num):
            if type(str_or_num) == float or type(str_or_num) == int:
                return math.isnan(float(str_or_num))
            elif str_or_num == None:
                return True
            elif type(str_or_num) == str:
                # TO DO: define a parameter to set the NaN strings directly to pandas
                if str_or_num.lower() == "nan":
                    return math.isnan(float(str_or_num))
                elif str_or_num.lower() == "na":
                    return True
                elif str_or_num.lower() == "n/a":
                    return True
                elif str_or_num.lower() == "--":
                    return True
                elif str_or_num.lower() == "":
                    return True
            else:
                # will always return True with NaN values
                return str_or_num != str_or_num

        def encode(value, encoder, var_name, info):
            if value in encoder[var_name]:
                return encoder[var_name][value]
            if is_missing(value):
                return np.nan
            else:
                # if value exists and is unknown
                if "categorical" in info:
                    # TO DO: decide if unknown categorical should be -1, 0, or NaN
                    return -1
                else:
                    return np.nan

        for var_name in self.to_custom_encode:
            data.loc[:,var_name] = data.loc[:,var_name].apply(encode, args=[self.custom_encoders, var_name, features_info[var_name]])

        for var_name in self.to_auto_encode:
            data.loc[:,var_name] = data.loc[:,var_name].apply(encode, args=[self.auto_encoders, var_name, features_info[var_name]])

        return data

    def __onehot_encode_existing_values(self, data, var_name):
        encoded_columns_sparse_matrix = self.onehot_encoder_models[var_name].transform(data.loc[:,var_name].values.reshape(-1,1))
        encoded_matrix = encoded_columns_sparse_matrix.toarray()

        incomplete_feature_names = self.onehot_encoder_models[var_name].get_feature_names()
        feature_names = [string.replace("x0",var_name) for string in incomplete_feature_names]

        data.drop(var_name, axis=1, inplace=True)

        encoded_columns = pd.DataFrame(encoded_matrix, columns=feature_names)
        merge = pd.concat([data, encoded_columns], axis=1)

        self.original_names[var_name] = feature_names

        return merge

    def __ordinal_encode_existing_values(self, data, var_name):
        filled_only = data.loc[:,var_name].dropna()
        indexes = filled_only.index.values
        
        reshaped_column = filled_only.values.reshape(-1,1)
        encoded_values = self.ordinal_encoder_models[var_name].transform(reshaped_column)
        indexed_values = pd.Series(encoded_values[:,0], index=indexes)

        data.loc[:,var_name].update(indexed_values)
        return data

    def fit_mice_model(self, data):
        data = data.where(pd.notnull(data), None)
        columns = data.columns
        filled_data = self.mice.fit(data)

    def replace_na(self, data):
        columns = data.columns
        filled_data = self.mice.transform(data)
        filled_df = pd.DataFrame(filled_data, columns=columns)
        return filled_df

    def fit_embedder_model(self, data, to_embed, target_name):
        for var_name in to_embed:
            var_cardinality = data[var_name].nunique()

            self.__fit_scaler_for_NN(data, var_name)
            scaled_var_np_array = self.__scale_for_NN(data, var_name)

            self.embedder_models[var_name] = Embedder(scaled_var_np_array, var_cardinality, data[target_name], self.seed)
            self.to_embed.append(var_name)

    def __fit_scaler_for_NN(self, data, var_name):
        self.NN_scaler_models[var_name] = MinMaxScaler(feature_range=(0, 1))
        self.NN_scaler_models[var_name].fit(data.loc[:,var_name].values.reshape(-1,1))

    def __scale_for_NN(self, data, var_name):
        scaled_var = self.NN_scaler_models[var_name].transform(data.loc[:,var_name].values.reshape(-1,1))
        return scaled_var

    def embed_categorical(self, data):
        for var_name in self.to_embed:
            scaled_var_np_array = self.__scale_for_NN(data, var_name)
            data = self.__embed_categories(data, var_name, scaled_var_np_array)
        return data

    def __embed_categories(self, data, var_name, scaled_var):
        embedded_feature = self.embedder_models[var_name].predict(scaled_var)
        embedded_feature_df = pd.DataFrame(embedded_feature)
        data.drop(var_name, axis=1, inplace=True)

        for column_name, column_data in embedded_feature_df.iteritems():
            new_column_name = var_name+"_"+str(column_name)
            data.loc[:,new_column_name] = column_data.copy()
            self.original_names[var_name].append(new_column_name)

        return data

    def normalize_distribution(self, data, target, dont_normalize, meta_data):
        for var_name in data.columns:
            skewness = data.loc[:,var_name].skew()
            original_name = self.__original_name(var_name)
            onehot_encoded = original_name in self.to_onehot_encode
            excluded = original_name in dont_normalize
            info = meta_data["features"][original_name] if var_name != target else meta_data["target"]
            binary = "binary" in info
            continuous = "continuous" in info
            if continuous and not excluded and not onehot_encoded and not binary and skewness > self.skewness_threshold:
                data.loc[:,var_name] = np.where(data.loc[:,var_name] > 0, np.log(data.loc[:,var_name].copy().astype('float64')), data.loc[:,var_name].copy())
        return data

    def fit_scaler_models(self, data, dont_scale, all_continuous):
        for var_name in data.columns:
            if self.__original_name(var_name) not in dont_scale:
                if self.__original_name(var_name) in all_continuous:
                    skewness = data[var_name].skew()
                    if skewness > self.skewness_threshold:
                        self.scaler_models[var_name] = MinMaxScaler(feature_range=(0, 1))
                    elif skewness <= self.skewness_threshold:
                        self.scaler_models[var_name] = StandardScaler()
                else:
                    self.scaler_models[var_name] = MinMaxScaler(feature_range=(0, 1))

                self.scaler_models[var_name].fit(data.loc[:,var_name].values.reshape(-1,1))

    def scale(self, data, dont_scale):
        for var_name in data.columns:
            if self.__original_name(var_name) not in dont_scale:
                data.loc[:,var_name] = self.scaler_models[var_name].transform(data.loc[:,var_name].copy().values.reshape(-1,1))
        return data

    def split(self, data, target):
        X_train, X_test, y_train, y_test = train_test_split(
            data.copy().drop(target, axis=1).copy(), data.loc[:,target].copy(),
            test_size=self.test_size,
            random_state=self.seed
        )
        return X_train, X_test, y_train, y_test

    def __original_name(self, name):
        """Find if the variable has suffered redimensioning
        
            name: string
                Current name of variable

            This function returns the parameter "name" if it is in fact the 
            original name, and the original name (pre redimensioning) if it isn't
        """
        original = name
        for original_name, new_names in self.original_names.items():
            if name in new_names:
                original = original_name
        return original