import numpy as np
from collections import deque

#between node & gate: for node: node.out == None, gate: node.out != None
# if it's a node, get its value or grad directly, otherwise, get gate.out
def getVal(node):
	if node.type == "node":
		return node.val
	else:
		return node.out.val
	

def updateGrad(node, grad):
	if node.type == "node":
		node.grad = grad
	else:
		node.out.grad = grad

def getGradient(node):
	if node.type == "node":
		return node.grad
	else:
		return node.out.grad

class multiplyGate():
	def __init__(self, x, y):
		self.label = "*"
		self.type = "gate"
		self.x = x
		self.y = y
		x.next = self
		y.next = self
		self.next = None
		self.out = Node(0,0)
		self.degree = int(x.type=="gate") + int(y.type=="gate") 

	#x and y are nodes or gates
	def forward(self):
		xVal = getVal(self.x)
		yVal = getVal(self.y)
		if np.isscalar(xVal) and np.isscalar(yVal):
			self.out.val = xVal*yVal
		else:
			self.out.val = np.matmul(yVal, xVal)
		return self.out

	def backward(self):
		xVal = getVal(self.x)
		yVal = getVal(self.y)
		zVal = self.out.grad
		if np.isscalar(xVal) and np.isscalar(yVal):
			dx = getGradient(self.x)
			dy = getGradient(self.y)
			updateGrad(self.x, dx + zVal * yVal)
			updateGrad(self.y, dy + zVal * xVal)

		else:
			self.x.grad += np.matmul(yVal.T,dz)
			self.y.grad += np.matmal(xVal.T,dz)

class sumGate():
	def __init__(self, x, y):
		self.label = "+"
		self.x = x
		self.y = y
		self.type = "gate"
		x.next = self
		y.next = self
		self.next = None
		self.out = Node(0,0)
		self.degree = int(x.type=="gate") + int(y.type=="gate")  

	def forward(self):
		xVal = getVal(self.x)
		yVal = getVal(self.y)
		self.out.val = xVal + yVal
		return self.out

	def backward(self):
		dz = self.out.grad
		dx = getGradient(self.x)
		dy = getGradient(self.y)
		updateGrad(self.x, dx + dz)
		updateGrad(self.y, dy + dz)


class maxGate():
	def __init__(self, x, y):
		self.label = "max"
		self.type = "gate"
		self.x = x
		self.y = y
		x.next = self
		y.next = self
		self.next = None
		self.out = Node(0,0)
		self.degree = int(x.type=="gate") + int(y.type=="gate") 

	def forward(self):
		self.out.val = max(getVal(self.x),getVal(self.y))
		return self.out

	def backward(self):
		dx = getGradient(self.x)
		dy = getGradient(self.y)
		dz = getGradient(self.out)
		if (getVal(self.x) > getVal(self.y)):
			updateGrad(self.x, dx + dz)
		else:
			updateGrad(self.y, dy + dz)


class sigmoidGate():
	def __init__(self, x):
		self.label = "sigmoid"
		self.type = "gate"
		self.x = x
		x.next = self
		self.next = None
		self.out = Node(0,0)
		self.degree = int(x.type=="gate")

	def forward(self):
		self.out.val = self.sigmoidFunc(getVal(self.x))
		return self.out

	def backward(self):
		dz = getGradient(self.out)
		updateGrad(self.x, (self.out.val*(1 - self.out.val))*dz)

	def sigmoidFunc(self, value):
		return 1.0/(np.exp(-1*value) + 1)

class softmaxLossGate():
	def __init__(self, x):
		self.label = "softmaxCELoss"
		self.type = "gate"
		self.next = None
		self.out = Node(0,0)
		self.x = x
		x.next = self
		self.degree = int(x.type=="gate")

	def forward(self):
		xVal = getVal(self.x)
		self.out.val = self.softmaxFunc(xVal)
		return self.out

	def backward(self): #self.out.grad = predicted y - x (need to be initialized elsewhere)
		updateGrad(self.x, self.out.grad)

	def forward(self,yVal): #self.out.grad = predicted y - x (need to be initialized elsewhere)
		xVal = getVal(self.x)
		self.out.val = np.log(self.softmaxFunc(xVal))*yVal
		return self.out

	def backward(self,yVal): #self.out.grad = predicted y - x (need to be initialized elsewhere)
		updateGrad(self.x, yVal - self.out.grad)

	def calculate(self,vec):
		c = -np.max(vec)
		vec = [np.exp(i+c) for i in vec]
		return vec/(1.0*np.sum(vec))

	def softmaxFunc(self,val):
		if len(val.shape) > 1:
			return np.array([calculate(i) for i in val])
		elif len(val.shape) == 1:
			return self.calculate(val)
		else:
			return 1

class softmaxCEGate():
	def __init__(self, x):
		self.label = "softmaxCEGate"
		self.type = "loss"
		self.next = None
		self.out = Node(0,0)
		self.x = x
		x.next = self
		self.degree = int(x.type=="gate")

	def forward(self,yVal): #self.out.grad = predicted y - x (need to be initialized elsewhere)
		xVal = getVal(self.x)
		self.out.val = np.log(self.softmaxFunc(xVal))*yVal
		return self.out

	def backward(self,yVal): #self.out.grad = predicted y - x (need to be initialized elsewhere)
		updateGrad(self.x, yVal - self.out.grad)

	def calculate(self,vec):
		c = -np.max(vec)
		vec = [np.exp(i+c) for i in vec]
		return vec/(1.0*np.sum(vec))

	def softmaxFunc(self,val):
		if len(val.shape) > 1:
			return np.array([calculate(i) for i in val])
		elif len(val.shape) == 1:
			return self.calculate(val)
		else:
			return 1

class Node():
	def __init__(self, val, grad, label=""):
		self.label = label
		self.type = "node"
		self.val = val
		self.out = None
		self.grad = grad
		self.next = None
		self.degree = 0

class Training():
	def __init__(self, gateList, weights):
		self.pipeline = self.topologicalSort(gateList)
		self.weights = weights

	def forwardTraining(self, y):
		for gate in self.pipeline:
			if gate.type == "gate":
				gate.forward()
			elif gate.type == "loss":
				gate.forward(y)

	def backwardTraining(self, y):
		for gate in self.pipeline[::-1]:
			if gate.type == "gate":
				gate.backward()
			elif gate.type == "loss":
				gate.backward(y)

	def gradientDescent(self, step_size):
		for weight in self.weights:
			weight.val += step_size*weight.grad

	def topologicalSort(self, gateList):
		count = len(gateList)
		if len(gateList) == 0:
			print "come on it's empty"
			return
		Queue = deque([node for node in gateList if node.degree ==0])
		order = []
		labels = []
		while Queue:
			node = Queue.popleft()
			order.append(node)
			labels.append(node.label)

			# In the typical topological sort, should be 
			# for neighbor in node.neighbors
			# just assume each node is connected to one component, and we only sort the gates here
			neighbor = node.next
			if neighbor:
				neighbor.degree -=1
				if neighbor.degree == 0:
					Queue.append(neighbor)
		if len(order) == count:
			print labels
			return order
		else:
			print "not valid topological order"
			return
