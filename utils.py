import cv2

import numpy as np

from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

def read_image(image_path):
	image = cv2.imread(image_path)
	return image

def read_numpy(file_path):
	image_array = np.load(file_path)
	return image_array

def resize_image(image, size=(64,64)):
	resized_image = cv2.resize(image, size)
	return resized_image

def write_image(image_array, image_name):
	file_dir = 'static/img/' + image_name
	print("fdile, {}".format(file_dir))
	cv2.imwrite(file_dir, image_array)
	return file_dir

def transform_numpy_to_tensor(numpy_array):
	return transform(numpy_array)