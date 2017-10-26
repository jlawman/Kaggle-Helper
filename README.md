# Kaggle-Helper
Tools to aid Kaggle Explorations

I have placed functions which I have written to help my Kaggle exploration.

Currently contains:

compare_columns(df1,df2):
    Compares the columns discrepencies between two dataframes

remove_outliers(old_df,number_of_std,columns="All",skip="None")
    Removes outliers from a dataframe.

run_classifiers(classifiers, prepared_features,labels,cross_validation=5)
    Runs for loop to test various classifiers on dataset.