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
import utils
import argparse



def main():
    pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='modelisation running')
	parser.add_argument('path', type=str, help='')
	parser.add_argument('separator',type=str, help='')
	parser.add_argument('target', type=str, help='')
	parser.add_argument('local_threshold', type=float, help='')
	parser.add_argument('threshold', type=float, help='')
	parser.add_argument('column_fill', type=str, help='')
	parser.add_argument('model', type=str, help='')

	#global variables for the different processing
	args = parser.parse_args()
	path  = args.path
	target = args.target
	separator = args.separator
	local_threshold = float(args.local_threshold)
	threshold = float(args.threshold)
	column_fill = args.column_fill
	model = args.model

	#calling the scaler and nlp model for the name entity elimination
	scaler = StandardScaler()
	StandardScaler(copy=True, with_mean=True, with_std=True)
	nlp = spacy.load("en_core_web_sm")	

	#reading the dataset
	df = utils.read_dataset(path,separator)
	#global analysis of the dataset
	utils.broad_analysis(df)
	utils.missing_values_table(df)
	features_list = df.columns
	#dropping duplicates in the dataset to avoid duplicated data points to have more importance than they really have
	df = df.drop_duplicates()
	#removing columns containing only one value, brining useless noise to the dataset
	df = utils.remove_unique_feature(df)
	#removing proper nouns from the dataset
	name_features = utils.remove_name(nlp,df)
	df = df.drop(name_features,axis=1)
	#visualizing correlation matrix (linear correlation only
	utils.visualise_correlation(df)
	#one hot encoding of categorical data
	df = utils.one_hot_encoder(df)
	#processing of NaNs values
	df = utils.missing_values(df,'drop')
	features = df.columns.tolist()
	del features[features.index(str(target))]
	#converting float values to log(min(x)+1) if the distribution is skewed 
	#this will aloow to correct distribution to be gaussian for better outliers removal
	for feature in features:
		if df[feature].dtypes == 'float64':
			if df[feature].skew() == 0:
				pass
			else:
				print(df.columns)
				df[feature] = df[feature].apply(utils.convert_to_log,args=[min(df[feature].values.tolist())])
	#removing outliers from the dataset
	outliers_removal(df)
	#removing columns that are colinear between them
	features_to_keep = utils.remove_colinar_features(target,local_threshold,df)
	label = df[target]
	df = df[features_to_keep]
	#APPLY PCA to the remaining features for noise removal and better discrimination
	nb_components = utils.PCA_generator(df,threshold)
	utils.pca_components(df, nb_components)
	scaler.fit(df)
	data = scaler.transform(df)
	pca = PCA(n_components=nb_components)
	principalComponents = pca.fit_transform(df)
	list_features = []
	list_features = utils.create_list_features(list_features,nb_components)
	principalDf1 = pd.DataFrame(data = principalComponents, columns = list_features)
	df = utils.create_optimal_dataset(list_features,principalDf1,label)
	X_train, X_test, y_train, y_test = train_test_split(df[list_features], df.label,train_size=0.9, test_size=0.1)
	#using tpot either for regression or classification to create the best model
	if model == 'regression':
		pipeline_optimizer = TPOTRegressor()
		pipeline_optimizer = TPOTRegressor(generations=10, population_size=20, cv=5,random_state=42, verbosity=2)
	elif model == 'classification':
		pipeline_optimizer = TPOTClassifier()
		pipeline_optimizer = TPOTClassifier(generations=10, population_size=20, cv=5,random_state=42, verbosity=2)

	pipeline_optimizer.fit(X_train, y_train)
	pipeline_optimizer.export('tpot_exported_pipeline.py')

