## Get to the part that really matters faster, the science!

PreSci aims to make the data analysis and feature engineering process
faster for predictive modeling.

### !!!IMPORTANT!!!
It is made for use with continuous and binary targets only!
Targets must be fed as numeric to PreSci!

#### Import it:
##### (from the folder directly above the presci folder)
    from presci.app.presci import PreSci

#### After instantiating it:
    presci = PreSci(data, target_name)

#### You can easily plot features of a given dataset with:
    presci.plot_features()

#### You can print a meta data dictionary to the console with:
    presci.print_meta_data()

#### You can print a correlation table with:
    presci.print_correlation()

#### You can also infer on the data after the transformations with:
    presci.post_plot_features()
    presci.post_print_meta_data()
    presci.post_print_correlation()

#### You can transform data for training with:
    presci.fit_transform()

#### And for predicting with:
    presci.transform(data)

On feature engineering, this package replaces rare labels on categorical data, 
encodes categorical (and discrete, if requested) data, replaces missing data using MICE, 
normalizes the distribution of data with high skewness, embeds data using NNs (if requested), 
and scales data, in order.

### !!!IMPORTANT!!!
If you should choose to ordinally encode your categorical data,and you have some criteria to 
do so, I strongly encourage you to encode according to your criteria before feeding the data 
into PreSci!

PreSci has an ordinal encoder and an auto-encoder, the ordinal encoder uses no criteria, 
the auto-encoder works as follows:

If categorical variables are fed to PreSci in string form, and not parametrized 
as variables to be encoded in some form, PreSci will auto-encode them in relation 
to the target value in case the target is continuous or binary, if target is neither,
PreSci will ordinally encode variable with no criteria!!!

You can also especically ask for the auto-encoder to be used using the "auto_encode"
parameter (works for discrete variables too!).

With continuous targets, variable will be labeled according to target mean, 
and with binary targets the variable will be labeled according to incidence of
highest target value, which is 1 (labels will be ordered by the ratio of target 
value for that label).

Any categorical label unknown to the auto-encoder will be labeled as 0!
Non-categorical unknown values will be left as NaN for the MICE model to predict them.

#### Auto Encoder Example: 
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
