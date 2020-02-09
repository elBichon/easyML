import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tpot import TPOTRegressor
from tpot import TPOTClassifier
import re
import spacy
from sklearn.preprocessing import StandardScaler



def read_dataset(name,separator):
    data = pd.read_csv(name, sep=separator)
    return data

def create_list_features(list_features,nb_components):
    i = 0
    while i < nb_components:
        list_features.append('PC'+str(i+1))
        i += 1 
    return(list_features)

def broad_analysis(data):
    print('shape of the dataset')
    print(data.shape)
    print('============================================================')
    print('============================================================')
    print('columns in the dataset')
    print(data.columns)
    print('============================================================')
    print('============================================================')
    print('infos on the dataset')
    i = 0
    features_list = df.columns
    while i < len(df.columns):
        print(features_list[i])
        print(df[str(features_list[i])].unique())
        i += 1
    print('============================================================')
    print('============================================================')
    print('infos on the type repartition')
    print(df.dtypes.value_counts())
    print('============================================================')
    print('============================================================')
    print(data.info())
    print('============================================================')
    print('============================================================')
    print('head')
    print(data.head())
    print('============================================================')
    print('============================================================')
    print('tail')
    print(data.tail())
    print('============================================================')
    print('============================================================')    
    print('null data')
    print(data.isnull().any())
    print('============================================================')
    print('============================================================')
    print('description')
    print(np.round(data.describe()))
    print('============================================================')
    print('============================================================')
    print('counting_values')    
    i = 0
    while i < len(features_list):
        print('feature '+features_list[i])
        df[features_list[i]].value_counts(dropna=False)
        i += 1
   # plt.boxplot(data)
   # plt.ylim(0,10)
    
def scatterplotting(data):
    pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde');
    
def visualise_correlation(data):
    correlation = data.corr()
    # display(correlation)
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
    
def visualise_specific_correlation(x_axis,y_axis):
    specific_data = df[[str(y_axis), str(x_axis)]]
    gridA = sns.JointGrid(x=x_axis, y=y_axis, data=specific_data, size=6)
    gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})
    gridA = gridA.plot_marginals(sns.distplot)
    
def categorical_correlation(category,y_axis):
    fig, axs = plt.subplots(ncols=1,figsize=(10,6))
    sns.barplot(x=str(category), y=str(y_axis), data=volatileAcidity_quality, ax=axs)
    title = str(category)+'VS'+str(y_axis)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()
    
def outliers_removal(data):
        # For each feature find the data points with extreme high or low values
    for feature in data.keys():
        if data[feature].dtype == 'float64':
        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(data[feature], q=25)
         # TODO: Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(data[feature], q=75)
        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
            interquartile_range = Q3 - Q1
            step = 1.5 * interquartile_range
        # Display the outliers
            print("Data points considered outliers for the feature '{}':".format(feature))
            display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])
    # OPTIONAL: Select the indices for data points you wish to remove
            outliers = []
    # Remove the outliers, if any were specified
            good_data = data.drop(data.index[outliers]).reset_index(drop = True)
        else:
            pass
    return good_data

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
def read_dataset(name,separator):
    data = pd.read_csv(name, sep=separator)
    return data

def one_hot_encoder(df):
    le = LabelEncoder()
    le_count = 0
    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                # Train on the training data
                le.fit(df[col])
                # Transform both training and testing data
                df[col] = le.transform(df[col])            
                # Keep track of how many columns were label encoded
                le_count += 1
    print('%d columns were label encoded.' % le_count)
    # one-hot encoding of categorical variables
    df = pd.get_dummies(df)
    print('Training Features shape: ', df.shape)
    return(df)

#def remove_colinear_data():
def remove_unique_feature(df):
    i = 0
    features_list = df.columns
    while i < len(features_list):
        if len(df[features_list[i]].unique()) == 1:
            print('dropping: ',features_list[i])
            df.drop(features_list[i], 1, inplace=True)
        i += 1
    return df
#def eliminate_low_importance():
#def perform_PCA()
    
def missing_values(df,policy):
    i = 0
    features_list = df.columns
    print(len(features_list))
    while i < len(features_list):
        if df[features_list[i]].isnull().sum() != 0:
            print('treating feature: ',features_list[i])
            if policy == 'drop':
                df = df[pd.notnull(df[features_list[i]])]
            elif policy == 'forwardfill':
                df[features_list[i]].fillna(method='ffill')
            elif policy == 'backwardfill':
                df[features_list[i]].fillna(method='bfill')
            elif policy == 'median_fill':
                df[features_list[i]].fillna(df[features_list[i]].mean())
        i += 1
    print('treatment is over')
    return df

def features_importance(data,information_import,target):
    data_np = data.astype(np.int32).values
    X = data_np[:,:-1]
    y = data_np[:,-1]
    model = ExtraTreesClassifier()
    model.fit(X,y)
# display the relative importance of each attribute
    sorted_features = np.sort(model.feature_importances_)[::-1].tolist()
    sum_info = 0
    i = 0
    while i < len(sorted_features):
        if sum_info < information_import:
            sum_info = sum_info+sorted_features[i]
        elif sum_info > information_import:
            break
        i += 1
    model = LogisticRegression()
# create the RFE model and select 3 attributes
    rfe = RFE(model, i)
    rfe = rfe.fit(X,y)
    # summarize the selection of the attributes
    rank = rfe.ranking_.tolist()
    i = 0
    features_index = []
    while i < len(rank):
        if rank[i] == 1:
            features_index.append(i)
        i += 1
    best_features = []
    for index in features_index:
        best_features.append(data.columns[index])
    best_features.append(target)
    print('most significant features', best_features)
    return best_features


def broad_analysis(data):
    print('shape of the dataset: ',data.shape)
    print(data.columns)
    print('First five rows')
    print(data.head(n=5))
    print('first last rows')
    print(data.tail(n=5))
    print('five random samples')
    print(data.sample(n=5))
    print('looking for null values')
    print(data.isnull().any())
    print('generic infos about the dataset')
    print(data.info())

def tendancy_indicators(dataframe,feature):
    print('average value mean = ',dataframe[feature].mean())
    print('middle value: mediane = ',dataframe[feature].median())
    print('value that appears the most: mode = ',dataframe[feature].mode())
    print('value of the variance: variance = ',dataframe[feature].var(ddof=0))
    print('value of the standard deviation: std = ',dataframe[feature].std(ddof=0))
    print('value of the skewnesse: skew = ',dataframe[feature].skew())
    if dataframe[feature].skew() == 0:
        print('symetrical distribution')
    elif dataframe[feature].skew() > 0:
        print('spreading to the left')
    else:
        print('spreading to the right')
    print('value of the kurtosis: kurtosis = ',dataframe[feature].kurtosis())
    if dataframe[feature].kurtosis() == 3:
        print('normal distribution')
    elif dataframe[feature].kurtosis() > 3:
        print('values are concentrated')
    else:
        print('values are not concentrated')
    
def data_relation(data,size):
    pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = size, diagonal = 'kde')
    plt.show()
    correlation = data.corr()
    plt.figure(figsize=size)
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
    
def specific_correlation(feature1,feature2): 
    #Visualize the co-relation between quantitative variables
    #Create a new dataframe containing only two columns to visualize their co-relations
    feature1_feature2 = data[[feature1, feature2]]
    #Initialize a joint-grid with the dataframe, using seaborn library
    gridA = sns.JointGrid(x=feature2, y=feature1, data=feature1_feature2, size=6)
    #Draws a regression plot in the grid 
    gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})
    #Draws a distribution plot in the same grid
    gridA = gridA.plot_marginals(sns.distplot)

def specific_correlation_quantitative(feature1,feature2,title,size):
    fig, axs = plt.subplots(ncols=1,figsize=size)
    feature1_feature2 = data[[feature1, feature2]]
    sns.barplot(x=feature1, y=feature2, data=feature1_feature2, ax=axs)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()


#PCA
def value_repetition(df):
    i = 0
    counter = len(df)
    data_count = {str(df[0]): 1}
    while i < counter:
        if str(df[i]) in data_count:
            data_count[str(df[i])] += 1
        else:
            data_count[str(df[i])] = 1  
        i += 1
    return(data_count)
#function that counts how often a same value is repeated in the dataset
#this function will return the result as a dictionnary giving for each distinct value in the dataset the number of times it appears

def data_distribution(rows):
    parameter = []
    value = []
    i = 0
    while i < len(rows):
        parameter.append(rows[i])
        value.append(len(df[rows[i]].value_counts()))
        l1, l2 = parameter,value
        output_dictionnary = dict(zip(l1, l2))
        i += 1
    return(output_dictionnary)


def data_distribution_plot(output_dictionnary, x_label, y_label ,graph_title):
    plt.bar(range(len(output_dictionnary)), output_dictionnary.values(), align='center',color='g')
    plt.xticks(range(len(output_dictionnary)), list(output_dictionnary.keys()))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(graph_title)
    plt.show()

def PCA_generator(df):
    i = 0
    j = 0
    size = len(df.columns)
    X = df.values
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    print('Covariance matrix \n%s' %cov_mat)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])
    tot = sum(eig_vals)
    var_exp_sorted = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp_sorted = np.cumsum(var_exp_sorted)
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(size, 4))
        plt.bar(range(size), var_exp_sorted, alpha=0.5, align='center', label='individual explained variance')
        plt.step(range(size), cum_var_exp_sorted, where='mid', label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    pca = PCA(n_components=size)
    pca.fit(X_std)
    print('variance ratio ',pca.explained_variance_ratio_) 

def pca_components(df, nb_components):
    X = df.values
    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)
    pca = decomposition.PCA(n_components=nb_components)
    pca.fit(X_scaled) 
    print (pca.explained_variance_ratio_)
    print (pca.explained_variance_ratio_.sum())
    X_projected = pca.transform(X_scaled)  
    pcs = pca.components_

def qualitative_representation(feature,x_axis):
    df[feature].value_counts(normalize=True).plot(kind='pie')
    plt.axis(x_axis) 
    plt.show() 
    df[feature].value_counts(normalize=True).plot(kind='bar')
    plt.show()

def quantitative_representation(feature1,feature2,bucket_value,sample_size):
    bin_numebr = 1 + math.log2(sample_size)
    print('bin_number: ',bin_number)
    df[feature1].value_counts(normalize=True).plot(kind='bar',width=0.1)
    plt.show()
    df[feature2].hist(normed=True)
    plt.show()
    df[df.feature2.abs() < bucket_value][feature2].hist(normed=True,bins=bin_number)
    plt.show()

def array_representation(feature):
    effectifs = df[feature].value_counts()
    modalites = effectifs.index 
    tab = pd.DataFrame(modalites, columns = [feature]) 
    tab["n"] = effectifs.values
    tab["f"] = tab["n"] / len(df) 


def create_set(df,target):
    X = df.drop(target,axis=1)
    y = df[target]
    y = df[target].values.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X = scaler.transform(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
def remove_colinar_features(target,local_threshold,df):
    corr_matrix = df.corr().abs()
    i = 0
    features_to_keep = []
    correlation_target = corr_matrix[target].values.tolist()
    while i < len(correlation_target):
        if correlation_target[i] >= local_threshold:
            features_to_keep.append(df.columns[i])
        i += 1
    print(features_to_keep)

    label = df[target]
    i = features_to_keep.index(target)
    del features_to_keep[i]
    print(features_to_keep)

    i = 0
    while i < len(features_to_keep):
        corr_matrix = df[features_to_keep].corr().abs()
        correlation_target = corr_matrix[features_to_keep[i]].values.tolist()
        j = 0
        while j < len(correlation_target):
            if float(correlation_target[j]) > 0.5 and float(correlation_target[j]) < 1.0:
                corr_matrix = df[features_to_keep].corr().abs()
                print('current feature '+features_to_keep[i])
                features_to_keep.remove(features_to_keep[j])
            j += 1
        i += 1

    features_to_keep = list(set(features_to_keep))
    return(features_to_keep)


    

def PCA_generator(df,threshold):
    i = 0
    j = 0
    size = len(df.columns)
    X = df.values
    X_std = StandardScaler().fit_transform(X)
# Clculating Eigenvectors and eigenvalues of Cov matirx
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
 #  Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
    for i in eig_pairs:
        print(i[0])
    tot = sum(eig_vals)
    var_exp_sorted = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp_sorted = np.cumsum(var_exp_sorted)
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(size, 4))
        plt.bar(range(size), var_exp_sorted, alpha=0.5, align='center', label='individual explained variance')
        plt.step(range(size), cum_var_exp_sorted, where='mid', label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    pca = PCA(n_components=size)
    pca.fit(X_std)
    components = pca.explained_variance_ratio_.tolist()
    i = 0
    nb_components = 0
    components_values = components[0]
    while i < len(components):
        if components_values < threshold:
            components_values = components_values+components[i+1]
            nb_components += 1
        else:
            break
    return nb_components+1


def pca_components(df, nb_components):
    X = df.values
    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)
    pca = decomposition.PCA(n_components=nb_components)
    pca.fit(X_scaled) 
    X_projected = pca.transform(X_scaled)  
    pcs = pca.components_
    
def label_encoding(features_list):
    if df[features_list[i]].dtypes == 'object':
        feature = list(set(df[features_list[i]].values.tolist()))
        le = preprocessing.LabelEncoder()
        le.fit(feature)
        feature = le.transform(df[features_list[i]].values.tolist()).tolist()
        df[features_list[i]] = feature
    else:
        pass
    
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def remove_name(nlp,df):
    columns_to_remove =  []
    for column in df.columns:
        print(column)
        if df[column].dtypes == 'object':
            if hasNumbers(str(df[column].values.tolist()[0]).lower()) == True:
                pass
            else:
                doc = nlp(re.sub("[^a-z]"," ",str(df[column].values.tolist()[0]).lower()))
                for token in doc:
                    if token.pos_ == 'PROPN' and token.tag_ == 'NNP' and  token.dep_ == 'compound':
                        columns_to_remove.append(column)
    return(list(set(columns_to_remove)))

def create_optimal_dataset(list_features,principalDf1,label):
    my_data = {}
    i = 0
    while i < len(list_features):
        my_data[list_features[i]] = principalDf1[list_features[i]].values.tolist()
        i += 1
    df = pd.DataFrame.from_dict(my_data)
    df.insert(0, "label", label.values.tolist(), True) 
    return df
