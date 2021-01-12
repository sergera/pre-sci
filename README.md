## Purpose

The purpose of this project is purely educational, I do not claim that
it a general solution for machine learning problems in any way, nor
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

Any failure to use this project in a Machine Learning problem will result
in experience, any victory in perplexity, please do not assume I am attempting
to present myself as anything other than I am: A student who likes coding, and
is curious about Machine Learning.

Or in the words of someone much much wiser than myself:

“I have not failed. I've just found 10,000 ways that won't work.”
Thomas A. Edison

## In other words

PreSci aims to study automation of data analysis plots and 
feature engineering for predictive modeling.

### !!!IMPORTANT!!!
PreSci is not meant to apply any transformations on the target!
It is made for use with continuous and boolean targets only!
Targets must be fed as numeric to PreSci!

#### On feature engineering, this package: 
1- Does transformations as written in the callack parameter (if there are any),
2- replaces rare labels on categorical data, 
3- encodes categorical (and discrete, if requested) data, 
4- replaces missing data using MICE, 
5- normalizes the distribution of continuous data with high skewness, 
6- embeds data using NNs (if requested), - this feature is still needs work
7- scales data.

#### Import it:
##### (from directory directly above the presci directory)
    from presci.presci import PreSci

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

### Encoders
If you should choose to encode your categorical data with some criteria to do so, 
I strongly encourage you to encode according to your criteria in the callback parameter!

Read the Docs for examples!

PreSci has an onehot encoder, an ordinal encoder, and an auto-encoder, 
the ordinal encoder uses no criteria, the auto-encoder works as follows:

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
