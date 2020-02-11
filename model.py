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

	args = parser.parse_args()
	path  = args.path
	target = args.target
	separator = args.separator
	local_threshold = float(args.local_threshold)
	threshold = float(args.threshold)
	column_fill = args.column_fill
	model = args.model

	scaler = StandardScaler()
	nlp = spacy.load("en_core_web_sm")	
	StandardScaler(copy=True, with_mean=True, with_std=True)

	df = utils.read_dataset(path,separator)
	utils.broad_analysis(df)
	utils.missing_values_table(df)
	features_list = df.columns
	df = df.drop_duplicates()
	df = utils.remove_unique_feature(df)
	name_features = utils.remove_name(nlp,df)
	df = df.drop(name_features,axis=1)
	utils.visualise_correlation(df)
	df = utils.one_hot_encoder(df)
	df = utils.missing_values(df,'drop')
	features_to_keep = utils.remove_colinar_features(target,local_threshold,df)
	label = df[target]
	df = df[features_to_keep]
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
	if model == 'regression':
		pipeline_optimizer = TPOTRegressor()
		pipeline_optimizer = TPOTRegressor(generations=3, population_size=20, cv=5,random_state=42, verbosity=2)
	elif model == 'classification':
		pipeline_optimizer = TPOTClassifier()
		pipeline_optimizer = TPOTClassifier(generations=10, population_size=20, cv=5,random_state=42, verbosity=2)

	pipeline_optimizer.fit(X_train, y_train)
	pipeline_optimizer.export('tpot_exported_pipeline.py')



