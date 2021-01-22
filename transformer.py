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

from .embedder.embedder import Embedder
from .commons import infer
from .commons import assure

class Transformer:
    """Assist in the data transformations of a simples Machine Learning pipeline

    Transformer will save models and data for variables in a DataFrame with the purpose of:
        - Replacing rare labels in categorical variables
        - Encoding variables
        - Replacing missing data
        - Scaling variables
    ;these functionalities require calling 2 different methods each

    In addition, it has some single method functionalities, like:
        - Removing outliers
        - Log transforming
        - Splitting dataset

    Read PyDocs below for more detailed instructions
    """
    def __init__(self, seed=0, missing_values=["","-","--","n/a","na","nan"]):
        """
        seed: int
            Number used to set the random state for the split, MICE model, and embedder model

        missing_values: list
            List of strings to be considered missing values
            This will only be used with the auto encoder and the custom encoder
        """
        assure.type_equals([int], seed, "seed")
        assure.type_equals([list], missing_values, "missing_values")


        self.seed = seed
        self.missing_values = missing_values

        # dict of variables and their frequent labels
        self._frequent_labels = {}

        # multiple imputation by chained equations model
        self._mice = IterativeImputer(max_iter=100, random_state=self.seed, sample_posterior=True, verbose=1, min_value=0)

        # dict of onehot encoder models for features with low cardinality
        self._onehot_encoder_models = {}
        self._to_onehot_encode = []

        # dict of ordinal encoder models for each categorical feature to be embedded
        self._ordinal_encoder_models = {}
        self._to_ordinal_encode = []

        # dict of auto encoders made in relation to target value according to it's type
        self._auto_encoders = {}
        self._to_auto_encode =[]

        # dict of custom encodings to be used
        self._custom_encoder = {}
        self._to_custom_encode = []

        # dict of categorical embedder NN models for each categorical variable
        self._embedder_models = {}
        self._to_embed = []

        # dict of scaler and encoder models specifically used for 
        # variables being fed to the Embedder NN
        self._NN_scaler_models = {}
        self._NN_encoder_models = {}

        # dict of scaler models for each variable 
        self._scaler_models = {}
        self._variable_order_MICE = []
    
    """
    Public Methods Section
    """

    def remove_outliers(self, data, continuous_vars, outlier_threshold=3):
        """
        Removes rows from dataset that contain outliers of the variables in "continuous_vars" 

        This method should only be used in the training dataset

        data: Pandas DataFrame
            DataFrame with continuous variables where outliers rows will be removed

        continuous_vars: list
            List of continuous variable names contained in the DataFrame to be searched
            for outliers

        outlier_threshold: float
            Values with z-score greater or equal to this value will be considered outliers

            In other words, values that are more than or equal to "this value" times the 
            standard deviation for the variable, are considered outliers

            This has a standard value of 3, according to the statistical rule of thumb

        returns: Pandas DataFrame
            Copy of input DataFrame with outlier rows removed

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_remove_outliers = ["continuous_var_1","continuous_var_2"]

            transformer = Transformer()

            dataset_with_outliers_removed = transformer.remove_outliers(dataset, to_remove_outliers)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")
        assure.type_equals([list], continuous_vars, "continuous_vars")
        assure.type_equals([float, int], outlier_threshold, "outlier_threshold")

        copy = data.copy()
        for var_name in continuous_vars:
            outliers = infer.outliers(copy.loc[:,var_name], outlier_threshold)
            for index in outliers.to_dict():
                copy = copy.drop(index, axis="index")
        copy.reset_index(inplace=True, drop=True)
        return copy

    def set_frequent_labels(self, data, rare_threshold=0.01):
        """
        data: Pandas DataFrame
            DataFrame only containing categorical vars which will have 
            rare labels replaced

        rare_threshold: float
            Labels with frequency smaller than or equal to this value 
            for a variable will be considered rare

            Default of threshold is 1%

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_replace_rare = dataset.loc[:,["categorical_var_1","categorical_var_2"]]

            transformer = Transformer()

            rare_threshold = 0.01
            transformer.set_frequent_labels(to_replace_rare, rare_threshold)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")
        assure.type_equals([float], rare_threshold, "rare_threshold")

        copy = data.copy()
        for var_name in copy.columns:
            frequent_labels = infer.frequent_labels(copy.loc[:,var_name], rare_threshold)
            self._frequent_labels[var_name] = frequent_labels

    def replace_rare_labels(self, data):
        """
        Replaces rare labels set with "set_frequent_labels" method with the string "Rare"

        data: Pandas DataFrame
            DataFrame containing (but not limited to) all variables set with 
            "set_frequent_labels" method

        returns: Pandas DataFrame
            Copy of input DataFrame with labels replaced by the string "Rare"

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_replace_rare = dataset.loc[:,["categorical_var_1","categorical_var_2"]]

            transformer = Transformer()

            transformer.set_frequent_labels(to_replace_rare)

            rare_replaced = transformer.replace_rare_labels(to_replace_rare)
            dataset.update(rare_replaced)
                OR
            dataset_with_rare_replaced = transformer.replace_rare_labels(dataset)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")
    
        copy = data.copy()
        for var_name in self._frequent_labels.keys():
            all_labels = copy[var_name].dropna().unique()
            for label in all_labels:
                if label not in self._frequent_labels[var_name]:
                    copy.loc[:,var_name] = copy.loc[:,var_name].replace({label: "Rare"})    
        return copy

    def fit_onehot_encoder(self, data):
        """
        Fits onehot encoder models for variables, and saves their names
        
        Variable names that were already saved will not be considered
        
        Onehot encoder will set all columns to 0 for unknown values
        
        data: Pandas DataFrame
            DataFrame only containing variables to be onehot encoded

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_onehot_encode = dataset.loc[:,["to_encode_1","to_encode_2"]]

            transformer = Transformer()

            transformer.fit_onehot_encoder(to_onehot_encode)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        to_onehot = list(set(data.columns) - set(self._to_onehot_encode))
        for var_name in to_onehot:
            if var_name not in self._to_onehot_encode:
                self._onehot_encoder_models[var_name] = OneHotEncoder(handle_unknown="ignore")
                self._onehot_encoder_models[var_name].fit(data.loc[:,var_name].dropna().values.reshape(-1,1))
                self._to_onehot_encode.append(var_name)

    def onehot_encode(self, data):
        """
        Encodes all variables that were set with the "fit_onehot_encoder" method

        data: Pandas DataFrame
            DataFrame containing (but not limited to) all variables included in 
            "fit_onehot_encoder" method

        returns: Pandas DataFrame
            Copy of input DataFrame with encoded variables

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_onehot_encode = dataset.loc[:,["to_encode_1","to_encode_2"]]

            transformer = Transformer()

            transformer.fit_onehot_encoder(to_onehot_encode)

            dataset_with_encoded_vars = transformer.onehot_encode(dataset)

            **Don't use "DataFrame.update()" since onehot encoder will create new columns
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        copy = data.copy()
        for var_name in self._to_onehot_encode:
            copy = self.__onehot_encode_existing_values(copy, var_name)

        return copy

    def fit_ordinal_encoder(self, data):
        """
        Fits encoder models for variables, and saves their names

        Variable names that were already saved will not be considered

        Ordinal encoder returns 0 for unknown values

        data: Pandas DataFrame
            DataFrame only containing variables to be ordinally encoded

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_ordinal_encode = dataset.loc[:,["to_encode_1","to_encode_2"]]

            transformer = Transformer()

            transformer.fit_ordinal_encoder(to_ordinal_encode)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        to_ordinal = list(set(data.columns) - set(self._to_ordinal_encode))
        for var_name in to_ordinal:
            self._ordinal_encoder_models[var_name] = OrdinalEncoder(handle_unknown="ignore")
            self._ordinal_encoder_models[var_name].fit(data.loc[:,var_name].dropna().values.reshape(-1,1))
            self._to_ordinal_encode.append(var_name)

    def ordinal_encode(self, data):
        """
        Encodes all variables that were set with the "fit_ordinal_encoder" method

        data: Pandas DataFrame
            DataFrame containing (but not limited to) all variables included in 
            "fit_ordinal_encoder" method

        returns: Pandas DataFrame
            Copy of input DataFrame with encoded variables

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_ordinal_encode = dataset.loc[:,["to_encode_1","to_encode_2"]]

            transformer = Transformer()

            transformer.fit_ordinal_encoder(to_ordinal_encode)

            encoded_vars = transformer.ordinal_encode(to_ordinal_encode)
            dataset.update(encoded_vars)
                OR
            dataset_with_encoded_vars = transformer.ordinal_encode(dataset)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        copy = data.copy()
        for var_name in self._to_ordinal_encode:
            copy = self.__ordinal_encode_existing_values(copy, var_name)

        return copy

    def set_custom_encoder(self, custom_encoder):
        """
        Saves custom encoded values for correspondent variable values

        Any variable value not included will be treated as unknown

        Variable names that were already saved will not be considered

        custom_encoder: {variable: {label: encoded_value}}
            Dictionary of encodings for variables, 
            first level keys are variable names to be encoded, 
            first level values are label dicts,
                second level keys are current values, and 
                second level values are encoded values to be aplied with the 
                "custom_encode" method

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            transformer = Transformer()
            transformer.set_custom_encoder({
                "to_encode_1": {"value1": 0, "value2": 1, "value3": 3}
                "to_encode_2": {"value1": 0, "value2": 1, "value3": 3}
            })
        """
        assure.type_equals([dict], custom_encoder, "custom_encoder")

        to_custom = list(set(custom_encoder.keys()) - set(self._to_custom_encode))
        self._custom_encoder = custom_encoder
        for variable in custom_encoder:
            self._to_custom_encode.append(variable)

    def custom_encode(self, data):
        """
        Encodes all variables that were set with the "set_custom_encoder" method

        data: Pandas DataFrame
            DataFrame containing (but not limited to) all variables set with 
            "set_custom_encoder" method

        returns: Pandas DataFrame
            Copy of input DataFrame with encoded variables

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            transformer = Transformer()
            transformer.set_custom_encoder({
                "to_encode_1": {"value1": 0, "value2": 1, "value3": 3}
                "to_encode_2": {"value1": 0, "value2": 1, "value3": 3}
            })

            dataset_with_encoded_vars = transformer.custom_encode(dataset)
            
            **Don't use "DataFrame.update()" since custom encoder will return NaN for unknown values
            and "update()" ignores NaN values
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        copy = data.copy()
        for var_name in self._to_custom_encode:
            copy.loc[:,var_name] = copy.loc[:,var_name].apply(self.__encode, args=[self._custom_encoder, var_name])
        return copy

    def fit_auto_encoder_continuous_target(self, data, target_name):
        """
        Saves encoded values to correspondent variable values

        This method only considers the target mean for variable values
        when deciding the encoded value

        Variable names that were already saved will not be considered

        data: Pandas DataFrame
            DataFrame only containing variables to be encoded and target variable

        target_name: string
            String with target variable name

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_auto_encode = dataset.loc[:,["to_encode_1","to_encode_2","target_name"]]

            transformer = Transformer()

            transformer.fit_auto_encoder_continuous_target(to_auto_encode, "target_name")
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")
        assure.type_equals([str], target_name, "target_name")

        to_auto = list(set(data.columns) - set(self._to_auto_encode) - set([target_name]))
        for var_name in to_auto:
            ordered_labels = data.copy().groupby([var_name])[target_name].mean().sort_values().index
            ordinal_labels = {k: i for i, k in enumerate(ordered_labels, 0)}
            self._auto_encoders[var_name] = ordinal_labels
            self._to_auto_encode.append(var_name)

    def fit_auto_encoder_boolean_target(self, data, target_name):
        """
        Saves encoded values to correspondent variable values

        This method only considers the ratio of variable values rows
        where the target equals 1 when deciding the encoded value

        Variable names that were already saved will not be considered

        data: Pandas DataFrame
            DataFrame only containing variables to be encoded and target variable

        target_name: string
            String with target variable name

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_auto_encode = dataset.loc[:,["to_encode_1","to_encode_2","target_name"]]

            transformer = Transformer()

            transformer.fit_auto_encoder_boolean_target(to_auto_encode, "target_name")
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")
        assure.type_equals([str], target_name, "target_name")

        to_auto = list(set(data.columns) - set(self._to_auto_encode) - set([target_name]))
        for var_name in to_auto:
            feature_labels = data[var_name].dropna().unique()
            feature_size = len(data[var_name].copy().dropna().index)
            max_target = data[target_name].unique().max()
            
            label_map = {}
            for label in feature_labels:
                label_size = len(data[(data.loc[:,var_name] == label)].index)
                occurences = len(data[(data.loc[:,var_name] == label) & (data.loc[:,target_name] == max_target)].index)
                label_map[label] = occurences / label_size

            self._auto_encoders[var_name] = label_map
            self._to_auto_encode.append(var_name)

    def auto_encode(self, data):
        """
        Encodes all variables that were set with the "fit_auto_encoder_continuous_target" or
        "fit_auto_encoder_boolean_target" methods

        data: Pandas DataFrame
            DataFrame containing (but not limited to) all variables included in the 
            "fit_auto_encoder_continuous_target" or "fit_auto_encoder_boolean_target" methods

        returns: Pandas DataFrame
            Copy of input DataFrame with encoded variables

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_auto_encode = dataset.loc[:,["to_encode_1","to_encode_2"]]

            transformer = Transformer()

            transformer.fit_auto_encoder_continuous_target(to_auto_encode, "target_name")
                OR
            transformer.fit_auto_encoder_boolean_target(to_auto_encode, "target_name")

            dataset_with_encoded_vars = transformer.auto_encode(dataset)
            
            **Don't use "DataFrame.update()" since auto encoder will return NaN for unknown values
            and "update()" ignores NaN values

        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        copy = data.copy()
        for var_name in self._to_auto_encode:
            copy.loc[:,var_name] = copy.loc[:,var_name].apply(self.__encode, args=[self._auto_encoders, var_name])

        return copy

    def fit_mice(self, data):
        """
        Fits MICE model for variables

        Saves order of variables for the model

        data: Pandas DataFrame
            DataFrame only containing variables to have it's missing values replaced

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            transformer = Transformer()

            to_replace_missing = dataset.loc[:,["to_replace_1","to_replace_2"]]

            transformer.fit_mice(to_replace_missing)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        self._variable_order_MICE = data.columns
        data = data.where(pd.notnull(data), None)
        columns = data.columns
        filled_data = self._mice.fit(data)

    def replace_missing(self, data):
        """
        Replaces missing data using trained MICE model

        It replaces data with the fitted variable order, and returns 
        the data with the inputted variable order

        data: Pandas DataFrame
            DataFrame containing (but not limited to) all variables included in 
            "fit_mice" method

        returns: Pandas DataFrame
            Copy of input DataFrame with replaced missing values

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            transformer = Transformer()

            to_replace_missing = dataset.loc[:,["to_replace_1","to_replace_2"]]

            transformer.fit_mice(to_replace_missing)

            replaced_vars = transformer.replace_missing(to_replace_missing)
            dataset.update(replaced_vars)
                OR
            dataset_with_replaced = transformer.replace_missing(dataset)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        copy = data.copy()
        
        fitted_order = copy.loc[:,self._variable_order_MICE]
        columns = fitted_order.columns
        filled_data = self._mice.transform(fitted_order)
        filled_df = pd.DataFrame(filled_data, columns=columns)

        copy.update(filled_df)
        return copy

    def fit_embedder(self, data, target_name):
        """
        1- Fits scaler model for the target
        2- Scales target
        3- Fits encoder models for the variables to be embedded  
        4- Encodes said variables
        5- Fits embedder models for variables 
        6- Saves their names

        Variable names that were already saved will not be considered

        data: Pandas DataFrame
            DataFrame only containing variables to be embedded and target variable

        target_name: string
            String with target variable name

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_embed = dataset.loc[:,["to_embed_1","to_embed_2","target_name"]]

            transformer = Transformer()

            transformer.fit_embedder(to_embed, "target_name")
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")
        assure.type_equals([str], target_name, "target_name")

        
        self.__fit_scaler_for_NN(data, target_name)

        to_embed = list(set(data.columns) - set(self._to_embed) - set([target_name]))
        for var_name in to_embed:
            copy = data.copy()
            var_and_target = copy.loc[:,[var_name, target_name]]
            existing = var_and_target.dropna()

            # scale target
            var_cardinality = existing[var_name].nunique()
            scaled_target_np_array = self.__scale_for_NN(existing, target_name)

            # encode variable for the embbeding NN layer
            self.__fit_ordinal_encoder_for_NN(existing, var_name)
            encoded_var_np_array = self.__ordinal_encode_for_NN(existing, var_name)

            self._embedder_models[var_name] = Embedder(encoded_var_np_array, scaled_target_np_array, self.seed)
            self._to_embed.append(var_name)

    def embed(self, data):
        """
        data: Pandas DataFrame
            DataFrame containing (but not limited to) all variables included in 
            "fit_embedder" method

        returns: Pandas DataFrame
            Copy of input DataFrame with embedded variables
            
            Embedded columns will only have values in the rows where the 
            original variable had values

            The rest of the rows will be filled with NaN

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_embed = dataset.loc[:,["to_embed_1","to_embed_2"]]

            transformer = Transformer()

            transformer.fit_embedder(to_embed, "target_name")

            dataset_with_embedded_vars = transformer.embed(dataset)
            
            **Don't use "DataFrame.update()" since embedder will create new columns
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        copy = data.copy()
        for var_name in self._to_embed:
            variable_only = copy.loc[:,var_name].to_frame()
            existing = variable_only.dropna()
            encoded_var_np_array = self.__ordinal_encode_for_NN(existing, var_name)
            existing = self.__embed_categories(existing, var_name, encoded_var_np_array)

            # drop embedded variable original column
            copy.drop(var_name, axis=1, inplace=True)
            # add embedded columns to data, filled with NaN
            for column in list(set(existing.columns) - set(copy.columns)):
                copy[column] = np.nan
            # update columns for the rows that had values
            copy.update(existing)

        return copy

    def log(self, data):
        """
        Logs variables in an attempt to normalize their distribution

        This is useful for continuous variables with skewed distribution

        data: Pandas DataFrame
            DataFrame only containing continuous variables to be logged

        returns: Pandas DataFrame
            Copy of input DataFrame with logged variables

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_log = dataset.loc[:,["to_log_1","to_log_2"]]

            transformer = Transformer()

            logged = transformer.log(to_log)
            dataset.update(logged)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        copy = data.copy()
        for var_name in data.columns:
            copy.loc[:,var_name] = np.where(copy.loc[:,var_name] > 0, np.log(copy.loc[:,var_name].astype('float64')), copy.loc[:,var_name])
        return copy

    def fit_minmax_scaler(self, data):
        """
        Fits MinMax scaler models for variables, and saves their names    

        Variable names that were already saved will not be considered

        MinMax scaler is preferred to for continuous variables with skewed distribution,
        or non-continuous variables

        data: Pandas DataFrame
            DataFrame only containing variables to be scaled

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_scale = dataset.loc[:,["to_scale_1","to_scale_2"]]

            transformer = Transformer()

            transformer.fit_minmax_scaler(to_scale)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        to_scale = list(set(data.columns) - set(self._scaler_models.keys()))
        for var_name in data.columns:
            self._scaler_models[var_name] = MinMaxScaler(feature_range=(0, 1))
            self._scaler_models[var_name].fit(data.loc[:,var_name].values.reshape(-1,1))

    def fit_standard_scaler(self, data):
        """
        Fits standard scaler models for variables, and saves their names    

        Variable names that were already saved will not be considered

        Standard scaler is preferred for normally distributed continuous variables

        data: Pandas DataFrame
            DataFrame only containing variables to be scaled

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_scale = dataset.loc[:,["to_scale_1","to_scale_2"]]

            transformer = Transformer()

            transformer.fit_standard_scaler(to_scale)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        to_scale = list(set(data.columns) - set(self._scaler_models.keys()))
        for var_name in data.columns:
            self._scaler_models[var_name] = StandardScaler()
            self._scaler_models[var_name].fit(data.loc[:,var_name].values.reshape(-1,1))

    def scale(self, data):
        """
        Scales variables with saved models

        data: Pandas DataFrame
           DataFrame containing (but not limited to) all variables included in the 
           "fit_minmax_scaler" and "fit_standard_scaler" methods

        returns: Pandas DataFrame
            Copy of input DataFrame with scaled variables

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            to_minmax_scale = dataset.loc[:,["to_minmax_1","to_minmax_2"]]
            to_standard_scale = dataset.loc[:,["to_standard_1","to_standard_2"]]

            transformer = Transformer()

            transformer.fit_minmax_scaler(to_minmax_scale)
            transformer.fit_standard_scaler(to_standard_scale)

            dataset_with_scaled_vars = transformer.scale(dataset)
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")

        copy = data.copy()
        for var_name in self._scaler_models.keys():
            copy.loc[:,var_name] = self._scaler_models[var_name].transform(copy.loc[:,var_name].values.reshape(-1,1))

        return copy

    def split(self, data, target_name, test_size=0.1):
        """
        Splits data for training

        data: Pandas DataFrame
            DataFrame with training dataset including all features and target for splitting

        target_name: string
            String with target name

        test_size: float
            Ratio of dataset that will be split for testing during training

        returns: splitted data

        EXAMPLE:
            import pandas as pd
            from presci.transformer import Transformer

            dataset = pd.read_csv("my_dataset.csv")

            transformer = Transformer()

            X_train, X_test, y_train, y_test = transformer.split(dataset, "target_name")
        """
        assure.type_equals([type(pd.DataFrame())], data, "data")
        assure.type_equals([str], target_name, "target_name")

        copy = data.copy()
        X_train, X_test, y_train, y_test = train_test_split(
            copy.drop(target_name, axis=1), copy.loc[:,target_name],
            test_size=test_size,
            random_state=self.seed
        )
        return X_train, X_test, y_train, y_test

    """
    Private Methods Section
    """

    def __encode(self, value, encoder, var_name):
        """
        Used for encoding in Transformer's proprietary encoding methods
        (namely auto_encode, and custom_encode)

        value: string or number
            Current variable value

        encoder: dict
            Dict that relates current variable values to encoded ones

        var_name: string
            String with variable name

        returns: encoded value
            Value represented in encoder dict if it exists, 
            NaN if it doesn't (is unknown), or if it already is NaN
        """
        def is_missing(str_or_num):
            """
            Finds if value is missing, NaN, or None

            str_or_num: string or number
               Value to be avaluated

            missing_values: list
                List with value to be considered missing values

            returns: boolean
                True if value is missing 
            """
            if type(str_or_num) == float or type(str_or_num) == int:
                return math.isnan(float(str_or_num))
            elif str_or_num == None:
                return True
            elif type(str_or_num) == str:
                # TO DO: define a parameter to set the NaN strings directly to pandas
                if str_or_num.lower() in self.missing_values:
                    return True
            else:
                # will always return True with NaN values
                return str_or_num != str_or_num

        if value in encoder[var_name]:
            return encoder[var_name][value]
        if is_missing(value):
            return np.nan
        else:
            # if value exists and is unknown
            return np.nan

    def __onehot_encode_existing_values(self, data, var_name):
        """
        data: Pandas DataFrame
            DataFrame with variable to be encoded

        var_name: string
            String with variable name

        returns: Pandas DataFrame
            DataFrame with variable in onehot encoded form
        """
        copy = data.copy()

        encoded_columns_sparse_matrix = self._onehot_encoder_models[var_name].transform(copy.loc[:,var_name].values.reshape(-1,1))
        encoded_matrix = encoded_columns_sparse_matrix.toarray()

        incomplete_feature_names = self._onehot_encoder_models[var_name].get_feature_names()
        feature_names = [string.replace("x0",var_name) for string in incomplete_feature_names]

        copy.drop(var_name, axis=1, inplace=True)

        encoded_columns = pd.DataFrame(encoded_matrix, columns=feature_names)
        merge = pd.concat([copy, encoded_columns], axis=1)

        return merge

    def __ordinal_encode_existing_values(self, data, var_name):
        """
        data: Pandas DataFrame
            DataFrame with variable to be encoded

        var_name: string
            String with variable name

        returns: Pandas DataFrame
            DataFrame with variable in ordinally encoded form
        """
        copy = data.copy()

        filled_only = copy.loc[:,var_name].dropna()
        indexes = filled_only.index.values
        
        reshaped_column = filled_only.values.reshape(-1,1)
        encoded_values = self._ordinal_encoder_models[var_name].transform(reshaped_column)
        indexed_values = pd.Series(encoded_values[:,0], index=indexes)

        copy.loc[:,var_name].update(indexed_values)
        return copy

    def __fit_scaler_for_NN(self, data, var_name):
        """
        Fits scaler models for variables to be inserted in the Neural Network

        data: Pandas DataFrame
            DataFrame with variables to be scaled

        var_name: string
            String with variable name to be scaled
        """
        self._NN_scaler_models[var_name] = MinMaxScaler(feature_range=(0, 1))
        self._NN_scaler_models[var_name].fit(data.loc[:,var_name].values.reshape(-1,1))

    def __scale_for_NN(self, data, var_name):
        """
        Scales variables before being inserted in the Neural Network

        data: Pandas DataFrame
            DataFrame with variables to be scaled

        var_name: string
            String with variable name to be scaled

        returns: numpy array
            Numpy array with scaled variable
        """
        scaled_var = self._NN_scaler_models[var_name].transform(data.loc[:,var_name].values.reshape(-1,1))
        return scaled_var

    def __fit_ordinal_encoder_for_NN(self, data, var_name):
        """
        Fits encoder models for variables to be inserted in the Neural Network

        data: Pandas DataFrame
            DataFrame with variables to be encoded

        var_name: string
            String with variable name to be encoded
        """
        self._NN_encoder_models[var_name] = OrdinalEncoder(handle_unknown="ignore")
        self._NN_encoder_models[var_name].fit(data.loc[:,var_name].values.reshape(-1,1))

    def __ordinal_encode_for_NN(self, data, var_name):
        """
        Encodes variables before being inserted in the Neural Network

        data: Pandas DataFrame
            DataFrame with variables to be encoded

        var_name: string
            String with variable name to be encoded

        returns: numpy array
            Numpy array with encoded variable
        """
        encoded_var = self._NN_encoder_models[var_name].transform(data.loc[:,var_name].values.reshape(-1,1))
        return encoded_var

    def __embed_categories(self, data, var_name, encoded_var):
        """
        data: Pandas DataFrame
            DataFrame with variable to be embedded

        var_name: string
            String with variable name to be embedded

        encoded_var: numpy array
            Numpy array with variable to be embedded

        returns: Pandas DataFrame
            DataFrame with embedded variable
        """
        copy = data.copy()

        embedded_feature = self._embedder_models[var_name].predict(encoded_var)

        # index has to be the same as copy, since copy could have dropped missing values
        # so that values fill the corret rows when assigning to copy
        new_index = pd.Index(copy.index.tolist())
        embedded_feature_df = pd.DataFrame(embedded_feature, index=new_index)

        copy.drop(var_name, axis=1, inplace=True)

        for column_name, column_data in embedded_feature_df.iteritems():
            new_column_name = var_name+"_"+str(column_name)
            copy.loc[:,new_column_name] = column_data

        return copy