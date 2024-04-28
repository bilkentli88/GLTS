import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

# Define the main dataset folder path
DATASET_MAIN_FOLDER = "Datasets"

def load_dataset(dataset_name):
	return load_data(DATASET_MAIN_FOLDER , dataset_name,extension="")



def load_data(folder, dataset,extension = ".txt"):
	train_file = "".join([dataset, "_TRAIN", extension])
	test_file = "".join([dataset, "_TEST",extension])

	train_file = os.path.join(folder, train_file)
	test_file = os.path.join(folder, test_file)

	train_data = np.loadtxt(train_file)
	test_data = np.loadtxt(test_file)

	train_features = (train_data[:, 1:] - train_data[:, 1:].mean(axis=1).reshape(-1,1))/train_data[:, 1:].std(axis=1).reshape(-1,1)
	test_features = (test_data[:, 1:] - test_data[:, 1:].mean(axis=1).reshape(-1, 1))/test_data[:, 1:].std(axis=1).reshape(-1,1)
	enc = OneHotEncoder(handle_unknown='ignore')
	enc.fit(train_data[:, 0].reshape(-1, 1))
	# 0,1,2 class values changed to below one hot encoding
	# OneHotEncoder -> Softmax duzgun calışsın diye
	# 0 -> 0 0 1
	# 1 -> 0 1 0
	# 2 -> 1 0 0

	train_labels = enc.transform(train_data[:, 0].reshape(-1, 1)).toarray()
	test_labels = enc.transform(test_data[:, 0].reshape(-1, 1)).toarray()

	return train_features, train_labels.astype(np.int32), test_features, test_labels.astype(np.int32)


def convert_to_bags_2(data, bag_size, stride_ratio):
	if isinstance(data, torch.Tensor):
		data = data.numpy()
	#print(type(data))
	#print(data.shape)
	#print(data.dtype)
	bag_size = int(bag_size)
	#print("bag_size",bag_size)
	#print("stride_ratio",stride_ratio)
	bags = []
	stride = int(max(round(stride_ratio*bag_size),1))
	for i in range(data.shape[0]):
		instance = []
		size = data[i].shape[0]
		window = int(bag_size)
		while True:
			instance.append(data[i][window-bag_size: window])
			window += stride
			if window >= size:
				window = size
				instance.append(data[i][window - bag_size: window])
				break
		#print(len(instance))
		bags.append(np.array(instance))
	return torch.from_numpy(np.array(bags)).float()

def get_bag_size(dataset_name,bag_ratio):
	train_file = "".join([dataset_name, "_TRAIN", ""])
	train_file = os.path.join(DATASET_MAIN_FOLDER, train_file)
	train_data = np.loadtxt(train_file)
	time_series_size = train_data.shape[1]-1
	bag_size = int(time_series_size*bag_ratio)
	return bag_size


if __name__ == "__main__":
	dataset_name = "BirdChicken"
	bag_ratio = 0.1
	
	bag_size = get_bag_size(dataset_name,bag_ratio)
	print(bag_size)
