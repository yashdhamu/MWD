import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from os import walk, path
import cv2
import torchvision
from torchvision import datasets, models, transforms 
from pathlib import Path
from torchvision.io import read_image, ImageReadMode


#Loading the dataset to access it using the id
dataset = torchvision.datasets.Caltech101('/Users/dhamu/Downloads/')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

# #The below block of code is added to overcome .DS_Store file error
# root_dir = '/Users/dhamu/Downloads/'
# # Define a custom dataset class to filter out the .DS_Store file
# class CustomCaltech101(torchvision.datasets.Caltech101):
# 	def find_classes(self, dir):
# 		classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d != '.DS_Store']
# 		classes.sort()
# 		class_to_idx = {classes[i]: i for i in range(len(classes))}
# 		return classes, class_to_idx
# # Create an instance of the custom dataset
# dataset = CustomCaltech101(root=root_dir, download=True)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)


def hog_fd(pil_image):
	pil_image = pil_image.convert('RGB') # Convert image to an RGB Image
	image = np.asarray(pil_image) # Convert image to an numpy object for easier processing
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Converting BGR to Grayscale
	resized_image = cv2.resize(gray_image, (300, 100)) #Resizing image as asked
	gradient_x = ndimage.sobel(resized_image, axis=1, mode='constant') # Calculating X gradient
	gradient_y = ndimage.sobel(resized_image, axis=0, mode='constant') # Calculating Y gradient
	gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2) #Calculating magnitude
	gradient_orientation = np.arctan2(gradient_y, gradient_x) * (180 / np.pi) #Calculating orientation(stongest edge)
	cell_size_x = resized_image.shape[1] // 10 # As grid is 10X10
	cell_size_y = resized_image.shape[0] // 10 # As grid is 10X10
	num_bins = 9 #Total bins to be considered
	histograms = [] #list to store histograms for individual cell

	for i in range(0, resized_image.shape[0], cell_size_y):
		for j in range(0, resized_image.shape[1], cell_size_x):
			#Calculating region of intrest(cell)
			cell_roi_x = j
			cell_roi_y = i
			cell_roi_width = cell_size_x
			cell_roi_height = cell_size_y
			#Calculating gradient and orientation for the ROI
			cell_roi = gradient_orientation[cell_roi_y:cell_roi_y+cell_roi_height, cell_roi_x:cell_roi_x+cell_roi_width]
			cell_magnitude = gradient_magnitude[cell_roi_y:cell_roi_y+cell_roi_height, cell_roi_x:cell_roi_x+cell_roi_width]
			histogram = np.zeros(num_bins) #initializing empty histogram
			for y in range(cell_roi.shape[0]):
				for x in range(cell_roi.shape[1]):
					bin_index = int((cell_roi[y, x] + 180) / 360 * num_bins) #Converting rad to dgeree and filling data for that degree
					histogram[bin_index] += cell_magnitude[y, x] #Constructing histogram for cell
			histograms.append(histogram)
	feature_descriptor = np.concatenate(histograms) #Concatenating all histograms for final histogram
	return feature_descriptor




def color_moments_fd(pil_image):
	pil_image = pil_image.convert('RGB')# Convert image to an RGB Image
	image = np.asarray(pil_image)# Convert image to an numpy object for easier processing
	resized_image = cv2.resize(image, (300, 100)) # Resizing image
	num_rows = 10 # As grid size is 10x10
	num_cols = 10 # As grid size is 10x10
	color_moments = [] #To capture color moment for a particular cell
	cell_height = resized_image.shape[0] // num_rows
	cell_width = resized_image.shape[1] // num_cols

	for i in range(num_rows):
		for j in range(num_cols):
			# Calculating region of intrest
			cell_roi_x = j * cell_width
			cell_roi_y = i * cell_height
			cell_roi_width = cell_width
			cell_roi_height = cell_height
			cell_roi = resized_image[cell_roi_y:cell_roi_y+cell_roi_height, cell_roi_x:cell_roi_x+cell_roi_width]
			channel_moments = [] #To store CMs for each cell for a particular channel
			for channel in range(3): #3 for 3 channels(B, G, R)
				channel_mean = np.mean(cell_roi[:, :, channel]) # Calculating mean
				channel_std = np.std(cell_roi[:, :, channel]) # Calculating Std Deviation
				channel_skewness = (np.mean((cell_roi[:, :, channel] - channel_mean) ** 3)) ** 1/3 # Calculating Skewness
				channel_moments.extend([channel_mean, channel_std, channel_skewness])
			color_moments.extend(channel_moments)

	feature_descriptor = np.array(color_moments)
	return feature_descriptor


def resnet_layer_fd(pil_image):
	resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) #Loading resnet modle with IMAGENET1KV2 weights
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
	]) #Image preprocessing
	pil_image = pil_image.convert('RGB')# Convert image to an RGB Image
	image = pil_image # Loading image
	image_tensor = transform(image).unsqueeze(0) #Model expects a batch, but we are passing single image thus adding extra dimension
	
	#Output to be stored for respective layer
	hook_output_avgpool = []
	hook_output_layer3 = []
	hook_output_fc_layer = []

	#Hook function created to capture output from that respective layer
	def hook_fn_avgpool(module, input, output):
		hook_output_avgpool.append(output)

	def hook_fn_layer3(module, input, output):
		hook_output_layer3.append(output)

	def hook_fn_fc_layer(module, input, output):
		hook_output_fc_layer.append(output)

	#Calling hook to capture output from the particular layer
	avgpool_layer = resnet_model.avgpool
	hook_avgpool = avgpool_layer.register_forward_hook(hook_fn_avgpool)

	layer3 = resnet_model.layer3
	hook_layer3 = layer3.register_forward_hook(hook_fn_layer3)

	fc_layer = resnet_model.fc
	hook_fc_layer = fc_layer.register_forward_hook(hook_fn_fc_layer)

	resnet_model.eval()

	with torch.no_grad(): # As we dont need to change parameters so no need to calculate gradient
		_ = resnet_model(image_tensor)

	# Removing hook previously added
	hook_avgpool.remove()
	hook_layer3.remove()
	hook_fc_layer.remove()

	avgpool_output = hook_output_avgpool[0] # As only one value is captured we need to extract 1st element
	layer3_output = hook_output_layer3[0] # As only one value is captured we need to extract 1st element
	fc_layer_output = hook_output_fc_layer[0] # As only one value is captured we need to extract 1st element

	# Removing extra dimension and getting output in required format
	reduced_output_avgpool = torch.mean(avgpool_output.view(1, -1, 2), dim=2).squeeze()
	reduced_output_layer3 = torch.mean(layer3_output.view(1024, 14, 14), dim=(1, 2)).squeeze()
	return reduced_output_avgpool, reduced_output_layer3, fc_layer_output

#Main Function
if __name__ == "__main__":
	print('*-*'*10)
	print('''1 -> COLOR MOMENTS
2 -> HOG
3 -> RESNET_AVGPOOL
4 -> RESNET_LAYER3
5 -> RESNET_FC_LAYER''')
	print('*-*'*10)
	print('Please select one of the feature model')
	fd_model = input()
	print('Please input the image ID')
	image_id = int(input())
	img, label = dataset[image_id] 
	if fd_model == '1':
		feature = color_moments_fd(img)
		print(f"Feature Model: Color Moments")
		print(f"Image ID: {image_id}")
		print(feature)
	elif fd_model == '2':
		feature = hog_fd(img)
		print(f"Feature Model: HOG")
		print(f"Image ID: {image_id}")
		print(feature)
	elif fd_model == '3':
		feature, _, _ = resnet_layer_fd(img)
		print(f"Feature Model: ResNet Avgpool")
		print(f"Image ID: {image_id}")
		print(feature)
	elif fd_model == '4':
		_, feature, _ = resnet_layer_fd(img)
		print(f"Feature Model: ResNet Layer 3")
		print(f"Image ID: {image_id}")
		print(feature)
	elif fd_model == '5':
		_, _ ,feature = resnet_layer_fd(img)
		print(f"Feature Model: ResNet FC Layer")
		print(f"Image ID: {image_id}")
		print(feature)
	else:
		print("Error: Invalid feature model specified.")
	img.show()







