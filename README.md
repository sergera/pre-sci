## Purpose

The purpose of this project is purely educational, I do not claim that
it is a general solution for machine learning problems in any way, nor
do I claim that it is the most performatic, scalable, best, or even the 
safest solution.

This project was born simply of my proactive curiosity for Machine Learning, 
It was literally started along with my first Machine Learning project and
it's true purpose is to allow me to investigate if there are a set of 
feature transformation practices that suit more than one project (in other
words, if there is any common ground in that part of a simple pipeline), and 
to address feature transformations and plotting in a modular manner.

I do this in hopes that trying to find the common ground will lead me to
a better understanding of Machine Learning, the types of problems and 
approaches that it is used for, of the different algorithms, and of the
technologies that are commonly used with it in Python. 

Any failure to use this project in a Machine Learning problem will converge
in experience, any victory in perplexity, please do not assume I am attempting
to present myself as anything other than I am: A student who likes coding, and
is curious about Machine Learning.

Or in the words of someone much much wiser than myself:

“I have not failed. I've just found 10,000 ways that won't work.”
Thomas A. Edison

## In other words

PreSci aims to assist in the data analysis and feature engineering
process for predictive modeling.

#### PreSci uses Pandas

    So start by importing Pandas and loading your dataset.

    import pandas as pd
    dataset = pd.read_csv("my_dataset.csv")

### Analysis

    The analyzer module can plot features and create meta-data about the dataset.

#### Import Analyzer:

    from presci.analyzer import Analyzer

#### After instantiating:

    analyzer = Analyzer(dataset, "target_name")

#### Easily plot features:

    analyzer.plot_features()

#### Easily print useful meta-data about the variables:

    analyzer.print_meta_data()

#### Use meta-data for more useful stuff:

    meta_data = analyzer.get_meta_data()

    categorical = dataset.loc[:,meta_data["categorical_features"]]

    transformer.set_frequent_labels(categorical)

    replaced_rare_labels = transformer.replace_rare_labels(categorical)

##### Which brings me to the transformer!

### Transformations

    The transformer module assists with handling transformations for a 
    simple Machine Learning pipeline.

    It handles removing outliers, log transforming, replacing rare labels in 
    categorical variables, encoding variables, replacing missing data, scaling 
    variables and splitting data for training.

#### Import Transformer:

    from presci.transformer import Transformer

#### After instantiating:

    transformer = Transformer()

#### Some transformations require 1 method call:
##### 1- Removing outliers

    to_remove_outliers = ["continuous_var_1","continuous_var_2"]

    dataset_with_outliers_removed = transformer.remove_outliers(dataset, to_remove_outliers)

##### 2- Log transforming

    to_log = dataset.loc[:,["variable_to_log_1","variable_to_log_2"]]

    logged = transformer.log(to_log)

    dataset.update(logged)

##### 3- Splitting data for training

    X_train, X_test, y_train, y_test = transformer.split(dataset, "target_name")

#### Some transformations require 2 method calls:

    first_method > second_method

#### Basically the API of those transformations follows 2 rules:

    1- All of the fit and set methods receive a DataFrame and fit/set for 
    all variables in the DataFrame

    2- All of the methods that follow them receive a DataFrame and only
    change variables inputed in the previous function

##### 1- Replacing rare labels (set_frequent_labels > replace_rare_labels)

    to_replace_rare = dataset.loc[:,["categorical_var_1","categorical_var_2"]]

    transformer.set_frequent_labels(to_replace_rare)

    dataset_with_rare_replaced = transformer.replace_rare_labels(dataset)

##### 2- Onehot encoding (fit_onehot_encoder > onehot_encode)

    to_onehot_encode = dataset.loc[:,["variable_to_encode_1","variable_to_encode_2"]]

    transformer.fit_onehot_encoder(to_onehot_encode)

    dataset_with_encoded_vars = transformer.onehot_encode(dataset)

##### 3- Ordinal encoding (fit_ordinal_encoder > ordinal_encode)

    to_ordinal_encode = dataset.loc[:,["variable_to_encode_1","variable_to_encode_2"]]

    transformer.fit_ordinal_encoder(to_ordinal_encode)

    dataset_with_encoded_vars = transformer.ordinal_encode(dataset)

##### 4- Custom encoding (set_custom_encoder > custom_encode)

    transformer.set_custom_encoder({
        "variable_to_encode_1": {"value1": 0, "value2": 1, "value3": 3}
        "variable_to_encode_2": {"value1": 0, "value2": 1, "value3": 3}
    })

    dataset_with_encoded_vars = transformer.custom_encode(dataset)

##### 5.a- Auto encoding to continuous target (fit_auto_encoder_continuous_target > auto_encode)

    to_auto_encode = dataset.loc[:,["variable_to_encode_1","variable_to_encode_2"]]

    transformer.fit_auto_encoder_continuous_target(to_auto_encode, "target_name")

    dataset_with_encoded_vars = transformer.auto_encode(dataset)

##### 5.b- Auto encoding to boolean target (fit_auto_encoder_boolean_target > auto_encode)

    to_auto_encode = dataset.loc[:,["variable_to_encode_1","variable_to_encode_2"]]

    transformer.fit_auto_encoder_boolean_target(to_auto_encode, "target_name")

    dataset_with_encoded_vars = transformer.auto_encode(dataset)

##### 6- Replacing missing data (fit_mice > replace_missing)

    to_replace_missing = dataset.loc[:,["variable_to_replace_1","variable_to_replace_2"]]

    transformer.fit_mice(to_replace_missing)

    dataset_with_replaced = transformer.replace_missing(dataset)

##### 7- Deep Embedding (fit_embedder > embed)

    to_embed = dataset.loc[:,["variable_to_embed_1","variable_to_embed_2"]]

    transformer.fit_embedder(to_embed, "target_name")

    dataset_with_embedded_vars = transformer.embed(dataset)

##### 8.a- MinMax scaling

    to_scale = dataset.loc[:,["variable_to_scale_1","variable_to_scale_2"]]

    transformer.fit_minmax_scaler(to_scale)

    dataset_with_scaled_vars = transformer.scale(dataset)

##### 9.b- Standard scaling

    to_scale = dataset.loc[:,["variable_to_scale_1","variable_to_scale_2"]]

    transformer.fit_standard_scaler(to_scale)

    dataset_with_scaled_vars = transformer.scale(dataset)

#### If there are any questions

    Please refer to the PyDocs of the modules for further details and instructions

#### And for any subject, contact me!

    +55 021 98459-8394

    https://www.linkedin.com/in/sergio-joselli-1301191bb/

    https://www.instagram.com/sergio.joselli/

#### And if you feel like rock, listen to our album!

    https://open.spotify.com/album/65oAcgzLzLTQvCqhYjRyBE


