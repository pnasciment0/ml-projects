# Name: Paulo Nascimento
# Code for 3D to 2D reduction. I collaborated with Peter Robicheaux on this assignment.

import numpy as np
#import matplotlib.pyplot as plt
import pylab as plt
import operator
import math

f = '3Ddata.txt'

# myPCA - performs principal component analysis on a file 
# @params - a file 
# @return - nothing, calls graphPCA to principaloduce a figure

def myPCA(file):
	#load text data into numpy array
	arr = np.loadtxt(file)

	#delete the column that has the labels
	vals = np.delete(arr, 3, 1)

	#take transpose for easy working
	vals = vals.T
	
	#construct vector of mean
	mean_x = np.mean(vals[0,:])
	mean_y = np.mean(vals[1,:])
	mean_z = np.mean(vals[2,:])
	mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
	#mean_vector = np.mean(vals, axis = 0)

	#create scatter matrix = covariance matrix * scaling factor
	scatter_matrix = np.zeros((3,3))
	for i in range(vals.shape[1]):
   		scatter_matrix += (vals[:,i].reshape(3,1) - mean_vector).dot((vals[:,i].reshape(3,1) - mean_vector).T)
   	
   	#compute eigenvalues and eigenvectors
	eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
	eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
   	
   	#sort by eigenvalues, take top 2 for reduction matrix
	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	reduction_matrix = np.hstack((eig_pairs[0][1].reshape(3,1),eig_pairs[1][1].reshape(3,1)))
	
	#acquire transformed matrix
	transformed = reduction_matrix.T.dot(vals)
	graphReduction(transformed, 'PCA', arr.T[3], 1, 2, 3, 4)
   	
# graphPCA - graph results based off a matrix and labels for coloring
# @params - a matrix in R^2 to be graphed, labels to color, 4 numbers to represent
#           which labels mean what colors
# @return - a figure
def graphReduction(matrix, meth, labels, g, y, b, r):
	final = []
	#append x values
	final.append(matrix[0])

	#append y values
	final.append(matrix[1])

	#append labels = {1,2,3,4}
	final.append(labels)

	final = (np.asarray(final)).T

	#classify (x,y) into red, blue, yellow, or green depending on label
	final_red = []
	final_blue = []
	final_yellow = []
	final_green = []
	for x in final:
		if x[2] == g:
			final_green.append(x[:2])
		if x[2] == y:
			final_yellow.append(x[:2])
		if x[2] == b:
			final_blue.append(x[:2])
		if x[2] == r:
			final_red.append(x[:2])

	final_red = (np.asarray(final_red)).T
	final_blue = (np.asarray(final_blue)).T
	final_green = (np.asarray(final_green)).T
	final_yellow = (np.asarray(final_yellow)).T

	#plot red, blue, yellow, and green
	plt.plot(final_red[0,:], final_red[1,:], 'ro')
	plt.plot(final_blue[0,:], final_blue[1,:], 'bo')
	plt.plot(final_green[0,:], final_green[1,:], 'go')
	plt.plot(final_yellow[0,:], final_yellow[1,:], 'yo')
	plt.savefig('graph' + meth + '.png')
	plt.close()
	#plt.show()

myPCA(f)

# =========================================================================
# =========================================================================
# ================================ ISOMAP =================================
# =========================================================================
# =========================================================================

# simple euclidean distance 
def euclideanDistance(x, y):
	points = zip(x, y)
	diff = [pow((a - b), 2) for (a, b) in points]
	return math.sqrt(sum(diff))

# returns the k-nearest neighbors in a dataset given a particular vector
def getNeighbors(values, vector, k):
	distances = []
	length = len(vector) - 1
	for x in range(len(values)):
		dist = euclideanDistance(vector, values[x])
		distances.append((values[x], dist))
	distances.sort(key = operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

#given a list of neighbors, returns a list of indices where one can find them
def getIndices(neighbors, arr):
	index = []
	for x in neighbors:
		temp = np.where(arr == x)[0]
		counts = np.bincount(temp)
		counts = np.argmax(counts)
		index.append(counts)
	return index

#constructs the KNN matrix where the diagonals are 0, neighbors have distances, and everything else is infinity
def kNN_matrix(arr):
	final_matrix = np.full((500,500), math.inf)
	for i in range(len(final_matrix)):
		for j in range(len(final_matrix)):
			if i == j:
				final_matrix[i,j] = 0

	for i in range(len(final_matrix)):
		n = getNeighbors(arr, arr[i], 11)
		x = getIndices(n, arr)
		for index in x:
			final_matrix[i, index] = euclideanDistance(arr[i], arr[index])
	final_matrix = np.minimum(final_matrix, final_matrix.T)
	return final_matrix

#compute shortest path matrix
def FloydWarshall(arr):
	for k in range(len(arr)):
		for i in range(len(arr)):
			for j in range(len(arr)):
				if (arr[i,j] > arr[i,k] + arr[k,j]):
					arr[i,j] = arr[i,k] + arr[k,j]
	return arr

#lot of weird shit here, this function applies the Gram decomposition of QDQ^T to FloydWarshall result
def MDS(arr):
	temp = kNN_matrix(arr)
	temp = FloydWarshall(temp)
	d = np.square(temp)
	n = len(arr)
	p = np.identity(n) - (1/n)*(np.full((n,n),1))
	res = (-1/2)*p.dot(d).dot(p)
	evals, evecs = np.linalg.eig(res)
	eig_pairs = [(np.abs(evals[i]), evecs[:,i]) for i in range(len(evals))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	evals = [x[0] for x in eig_pairs]
	evecs = [x[1] for x in eig_pairs]
	evals = np.asarray(evals)
	evecs = np.asarray(evecs).T
	new_matrix = np.zeros(500)
	new_matrix[0] = evals[0]
	new_matrix[1] = evals[1]
	new_matrix = np.diagflat(new_matrix)
	new_matrix = np.sqrt(new_matrix)
	reduction = evecs.dot(new_matrix)
	final = [r[0:2] for r in reduction]
	final = np.asarray(final)
	return final

#finally, load in the dataset and apply Isomap to it
def myIsomap(file):
	arr = np.loadtxt(file)
	vals = np.delete(arr, 3, 1)
	transformed = MDS(vals).T
	graphReduction(transformed, 'ISO', arr.T[3], 1, 2, 3, 4)

myIsomap(f)




