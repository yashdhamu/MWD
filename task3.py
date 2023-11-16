import torch
import torchvision
import csv, os
# from task1 import hog_fd, color_moments_fd, resnet_layer_fd
import matplotlib.pyplot as plt
import cv2, re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import pandas as pd
from PIL import Image

#Loading the dataset to access it using the id
dataset = torchvision.datasets.Caltech101('/Users/dhamu/Downloads/')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

#To convert tensor to numpy
def parse_tensor(tensor_str):
	values = re.findall(r'-?\d+\.\d+', tensor_str)
	tensor = np.array(values, dtype=float)
	return tensor

def cosine_similarity(vector_a, vector_b):
	# Calculated the dot product of the two vectors
	dot_product = np.dot(vector_a, vector_b)

	# Calculated the Euclidean norm of each vector
	norm_a = np.linalg.norm(vector_a)
	norm_b = np.linalg.norm(vector_b)

	# Calculated the cosine similarity
	similarity = dot_product / (norm_a * norm_b)

	return similarity

if __name__ == "__main__":
	#Queries to be tested
	queries_id = [0, 880, 2500, 5122, 8676]
	#Load CSV stored to a variable as pandas object
	features_df = pd.read_csv('FD_Objects_all.csv')
	# number of similar objects to be extracted
	print('Please input value of K')
	k = int(input())
	#Exatch Pandas series for each feature
	# cm_features_df = features_df['ColorMoments']
	# hog_features_df = features_df['HOG']
	# rs1_features_df = features_df['ResNet_AvgPool_1024']
	# rs2_features_df = features_df['ResNet_Layer3_1024']
	rs3_features_df = features_df['ResNet_FC_1000']

	#Convert tensors to numpy in whole Data frane
	# cm_features_df = cm_features_df.apply(parse_tensor)
	# hog_features_df = hog_features_df.apply(parse_tensor)
	# rs1_features_df = rs1_features_df.apply(parse_tensor)
	# rs2_features_df = rs2_features_df.apply(parse_tensor)
	rs3_features_df = rs3_features_df.apply(parse_tensor)

	#removing NaN valuse as they will cause problem while calculating similarity
	# cm_features_df = cm_features_df.dropna()
	# cm_features_df = np.vstack(cm_features_df.values)
	# #Concatenating values to get single vector
	# hog_features_df = np.concatenate(hog_features_df.values).reshape(hog_features_df.shape[0], -1)

	top_k_indices_cm = []
	top_k_indices_hog = []
	top_k_indices_rs1 = []
	top_k_indices_rs2 = []
	top_k_indices_rs3 = []

	for image_id in queries_id:
		img, label = dataset[image_id]

		# feature_query_cm = cm_features_df[image_id]
		# feature_query_cm = np.array(feature_query_cm)
		# similarities_cm = [cosine_similarity(feature_query_cm, vector) for vector in cm_features_df]
		# similarities_cm = np.array(similarities_cm)
		# #Top K indices after sorting in reverse because cosine similarity should be as low as possible
		# top_k_indices_cm.append(np.argsort(similarities_cm)[::-1][:k])
		# #similaritu for HOG
		# feature_query_hog = hog_features_df[image_id]
		# similarities_hog = [cosine_similarity(feature_query_hog, vector) for vector in hog_features_df]
		# similarities_hog = np.array(similarities_hog)
		# top_k_indices_hog.append(np.argsort(similarities_hog)[::-1][:k])
		# #similaritu for ResNet
		# features_query_rs1 = rs1_features_df[image_id]
		# features_query_rs2 = rs2_features_df[image_id]
		features_query_rs3 = rs3_features_df[image_id]
		# features_query_rs1 = np.array(features_query_rs1)
		# features_query_rs2 = np.array(features_query_rs2)
		features_query_rs3 = np.array(features_query_rs3)
		# similarities_rs1 = [cosine_similarity(features_query_rs1, vector) for vector in rs1_features_df]
		# similarities_rs2 = [cosine_similarity(features_query_rs2, vector) for vector in rs2_features_df]
		similarities_rs3 = [cosine_similarity(features_query_rs3, vector) for vector in rs3_features_df]
		# similarities_rs1 = np.array(similarities_rs1)
		# similarities_rs2 = np.array(similarities_rs2)
		similarities_rs3 = np.array(similarities_rs3)
		# top_k_indices_rs1.append(np.argsort(similarities_rs1)[::-1][:k])
		# top_k_indices_rs2.append(np.argsort(similarities_rs2)[::-1][:k])
		top_k_indices_rs3.append(np.argsort(similarities_rs3)[::-1][:k])

	#creating director structure for storing the results
	base_folder_path = "Output"

	# Check if the base folder exists, if not, create it
	if not os.path.exists(base_folder_path):
		os.makedirs(base_folder_path)

	# Define the methods
	# methods = ["CM","HOG", "RS1", "RS2","RS3"]
	methods = ["RS3"]

	# Define the number of queries
	num_queries = len(top_k_indices_rs3)

	# Loop through each feature
		
	for i in range(1, num_queries+1):
		query_folder_path = os.path.join(base_folder_path,f"Query{i}")
		
		if not os.path.exists(query_folder_path):
			os.makedirs(query_folder_path)
		
		for subfolder in methods:
			subfolder_path = os.path.join(query_folder_path, subfolder)
			method_indices = locals()[f"top_k_indices_{subfolder.lower()}"]
			indices = method_indices[i-1]
			
			if not os.path.exists(subfolder_path):
				os.makedirs(subfolder_path)
			
			for j, index in enumerate(indices):
				pil_image, label = dataset[index]
				file_path = os.path.join(subfolder_path, f"image_{i}_{j}.jpg")
				print(file_path, index)
				pil_image.save(file_path)
				# print(f"Image saved at: {file_path}")








