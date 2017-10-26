
#################################
########### EDA #################
#################################

def compare_columns(df1,df2):
    """
    Compares the columns discrepencies between two dataframes

    Parameters:
    df1,df2 = two seperate dateframes

    Returns:
    Tuple: (list of columns in df1 but not in df2, list of columns in df2 but not in df1)

    Prints:
    List of columns in df1 but not in df2, list of columns in df2 but not in df1
    """
    df1_but_not_df2 =[column for column in df1.columns if column not in df2.columns]
    df2_but_not_df1 =[column for column in df2.columns if column not in df1.columns]
    print("Columns in df1 but not in df2 {}".format(df1_but_not_df2))
    print("Columns in df2 but not in df1 {}".format(df2_but_not_df1))

    return (df1_but_not_df2,df2_but_not_df1)
    

############################################
########### Data Cleaning #################
############################################

def remove_outliers(old_df,number_of_std,columns="All",skip="None"):
    """
    Removes outliers from a dataframe.
    
    Parameters:
    old_df: Series or dataframe
    
    number_of_std: Number of standard deviations for threshhold. 
                   Function will remove all outliers beyond this many standard deviations.
                   
    columns: The columns upon which the operation will be performed. (List of column names)
    
    skip: List of columns to be skipped.
    
    Returns:
    A dataframe with the outliers removed.
    
    """
    
    if isinstance(old_df,pd.core.series.Series): #If series passed, then only 
        current_series = old_df #set current series
        
        mean = np.mean(current_series)    #Mean
        std = np.std(current_series)      #Std
        threshold = number_of_std*std     #Threshhold = number of std * std
        
        new_df = old_df[np.abs(current_series-mean)<threshold] #Remove outliers from series
    else:
        if columns=="All": #Set columns
            columns=old_df.columns
            
        if skip!="None": #Skip any columns to be skipped
            columns = [name for name in list(old_df.columns) if name not in skip]
        
        for column in columns:
            current_series = old_df[column] #Iterate through each column

            mean = np.mean(current_series) #Set up threshold for which x should be within
            std = np.std(current_series)
            threshold = number_of_std*std

            new_df = old_df[np.abs(current_series-mean)<threshold] #Remove outliers from this column
    
    return new_df


############################################
########### Classification #################
############################################
def run_classifiers(classifiers, prepared_features,labels,cross_validation=5):
    """
    Runs for loop to test various classifiers on dataset.

    Parameters:
    classifiers = a list of initialized classifiers
    prepared_features = cleaned and standardized features
    labels = labels for classifications
    cross_validation = cross validation value

    Returns:
    The classifier which performed the best

    Prints:
    Training score for each classifier
    """
    highest_score=0
    best_classifier=""
    best_classifier_training_score=0

    for clf in classifiers:
        clf.fit(prepared_features,labels)

        pred = clf.predict(prepared_features)

        scores = cross_val_score(clf, prepared_features, labels, cv=cross_validation)

        training_score=clf.score(prepared_features,labels)
        cv_score_mean = scores.mean()

        if cv_score_mean>highest_score:
            highest_score=cv_score_mean
            best_classifier=clf
            best_classifier_training_score=training_score

        print("\n{}:\n\n\tTraining Score: {} \n\tCV Scores: {}".format(clf,training_score,cv_score_mean))

    print('\n\n----------------\nBest estimator:{}\n\n\tTraining Score: {} \n\tCV Scores: {}'.format(best_classifier,highest_score,best_classifier_training_score))
    return best_classifier