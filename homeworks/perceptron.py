# Name: Paulo Nascimento
# Code for perceptron. I collaborated with Peter Robicheaux on this assignment.

import numpy as np
import
import pylab as plt
import operator
import math

'''
load in all the data sets, train, todo, and labels
line techniques to get them into an array form
'''
train = 'train35.digits'
todo = 'test35.digits'
labels = 'train35.labels'

labels_list = []

with open(labels) as f:
	for line in f:
		labels_list.append(int(line))

labels_list = np.asarray(labels_list)

with open(train) as f:
	digits = f.readlines()

digits = [line.rstrip() for line in digits]
digits = [line.split() for line in digits]
digits = [[int(entry) for entry in line] for line in digits]
digits = np.asarray(digits)

with open(todo) as f:
	to_test = f.readlines()

to_test = [line.rstrip() for line in to_test]
to_test = [line.split() for line in to_test]
to_test = [[int(entry) for entry in line] for line in to_test]
to_test = np.asarray(to_test)

#cross-validation? haha, we pick 5 to smooth out the batch	
M = 5

weights = np.zeros(784)
errors = np.zeros(10000)

def perceptron(weights, digits, labels, errors, it):
	for i in range(len(digits)):
		if weights.dot(digits[i]) >= 0:
			prediction = 1
		else:
			prediction = -1
		if prediction == -1 and labels[i] == 1:
			errors[2000 * it + i] += 1
			weights = weights + digits[i]
		elif prediction == 1 and labels[i] == -1:
			errors[2000 * it + i] += 1
			weights = weights - digits[i]	
	return weights, errors

for i in range(M):
	weights, new_errors = perceptron(weights, digits, labels_list, errors, i)

final_errors = np.zeros(10000)
for i in range(10000):
	final_errors[i] = np.sum(new_errors[0:i+1])

plt.plot(range(10000), final_errors, color = 'black')

plt.savefig('errors.png')

def save_predictions(weights, digits):
	text_file = open("test35.predictions", "w")
	for i in range(len(digits)):
		pred = weights.dot(digits[i])
		if pred >= 0:
			text_file.write(str(1) + "\n")
		else:
			text_file.write(str(-1) + "\n")
	text_file.close()

save_predictions(weights, to_test)

