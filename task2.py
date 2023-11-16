import torch
import torchvision
import csv
from task1 import hog_fd, color_moments_fd, resnet_layer_fd #importing functions to generate FDs

#Loading the dataset to access it using the id
dataset = torchvision.datasets.Caltech101('/Users/dhamu/Downloads/')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

with open('FD_Objects.csv','w', newline='') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow(['ImageID', 'ColorMoments', 'HOG', 'ResNet_AvgPool_1024', 'ResNet_Layer3_1024', 'ResNet_FC_1000'])
	for image_ID in range(8677):
		img, label = dataset[image_ID]
		color_moments = color_moments_fd(img) #calling function to generate color moments for image
		color_moments = torch.from_numpy(color_moments) #Convert numpy to tensor
		hog_feature = hog_fd(img) #calling function to generate HOG for image
		hog_feature = torch.from_numpy(hog_feature) #Convert numpy to tensor
		resnet_avgpool, resnet_layer3, resnet_fc = resnet_layer_fd(img) #calling function to capture ResNet output for 3 laters for image
		torch.set_printoptions(profile="full") # Added to store whole tensor to CSV
		writer.writerow([image_ID, color_moments, hog_feature, resnet_avgpool, resnet_layer3, resnet_fc]) #Storage
		print(f"Saved feature descriptors for {image_ID}")

print("Feature extraction and CSV creation completed.")
