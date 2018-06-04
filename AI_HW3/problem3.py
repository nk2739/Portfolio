'''
Neil Kumar(nk2739)
AI Assignment 3
Programming Problem 3
4/15/18
'''
import pandas
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import matplotlib.image as mpimg

# Source: http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py

# Creates an image from the original image and the new labels 
def recreate_image(codebook,labels,width,height):
	dimensions = codebook.shape[1]

	image = np.zeros((width,height,dimensions))

	label_index = 0

	for i in range(width):
		for j in range(height):
			image[i][j] = codebook[labels[label_index]]
			label_index += 1

	return image

# Runs K-Means on images for the given value of K. 
def runKMeans(file,k):

	# Opens the given PNG image 
	trees = mpimg.imread(file)

	# Converts the PNG file into a 3D array to work with its RGB values
	trees = np.array(trees,dtype=np.float64) / 255

	# Extracts the image's dimensions
	width, height,dimensions = tuple(trees.shape)

	# Reshapes the image to a 2D array 
	image_array = np.reshape(trees,(width*height,dimensions))

	image_array_sample = shuffle(image_array,random_state=0)[:1000]

	# Creates a K-Means predictor from a sample of the image 
	kMeans = KMeans(n_clusters=k,random_state=0).fit(image_array_sample)

	# Predicts the labels of each pixel from the K-Means predictor
	labels = kMeans.predict(image_array)
	
	# Creates and displays the new image based on the new labels 
	plt.figure(1)
	plt.clf()
	ax = plt.axes([0,0,1,1])
	plt.axis('off')
	plt.title('Quantized Image('+str(k)+' Colors, K-Means)')
	plt.imshow(255*recreate_image(kMeans.cluster_centers_,labels,width,height))
	plt.show()

def main():

	inputFile = 'trees.png'
	k = 4
	runKMeans(inputFile,k)

if __name__ == '__main__':	
	main()