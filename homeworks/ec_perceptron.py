# Name: Paulo Nascimento
#extra credit perceptron??
import numpy as np
import pylab as plt
import operator
import math

'''
load in all the data sets, train, todo, and labels
line techniques to get them into an array form
'''
train = 'train01234.digits'
todo = 'test01234.digits'
labels = 'train01234.labels'

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

weights_m = []

for i in range(5):
	weights_m.append(np.zeros(784))

weights_m = np.asarray(weights_m)

def mc_perceptron(weights, digits, labels):
	for i in range(len(digits)):
		values = []
		values.append((0, weights[0].dot(digits[i])))
		values.append((1, weights[1].dot(digits[i])))
		values.append((2, weights[2].dot(digits[i])))
		values.append((3, weights[3].dot(digits[i])))
		values.append((4, weights[4].dot(digits[i])))
		most = max(values, key = lambda x: x[1])
		prediction = most[0]
		if prediction != labels[i]:
			#print("prediction: " + str(prediction) + "   label: " + str(labels[i]))
			weights[prediction] = weights[prediction] - (digits[i]/2)
			weights[labels[i]] = weights[labels[i]] + (digits[i]/2)
	return weights

for i in range(10):
	weights_m = mc_perceptron(weights_m, digits, labels_list)

def save_predictions(weights, digits):
	text_file = open("test01234.predictions", "w")
	for i in range(len(digits)):
		values = []
		values.append((0, weights[0].dot(digits[i])))
		values.append((1, weights[1].dot(digits[i])))
		values.append((2, weights[2].dot(digits[i])))
		values.append((3, weights[3].dot(digits[i])))
		values.append((4, weights[4].dot(digits[i])))
		most = max(values, key = lambda x: x[1])
		prediction = most[0]
		text_file.write(str(prediction) + "\n")
	text_file.close()

save_predictions(weights_m, to_test)



