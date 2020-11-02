import numpy as np 

#Sigmoid Function
def sigmoid(x, der=False):
	if der:
		return x * (1 - x)
	return 1 / (1+ np.exp(-x))

#Training Data Block/Input
x = np.array([[0, 0, 1],
              [1, 1 ,1], 
              [1, 0, 1], 
              [0, 1, 1]])
#Expected Training Results/Output
y = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
#Weights Creation
weights = 2 * np.random.random((3, 1)) - 1

l1 = []
#Iter = number of training cycles
#Calculations
for iter in range(20000):
	l0 = x
	l1 = sigmoid(np.dot(l0, weights))

	l1_error = y - l1

	l1_delta = l1_error * sigmoid(l1, True)

	weights += np.dot(l0.T, l1_delta)

print("Output after the training:")

print(l1)
#NN is Trained
new = np.array([1, 1, 0])
l1_new = sigmoid(np.dot(new, weights))

print("New result:")
print(l1_new)