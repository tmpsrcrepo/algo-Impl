import numpy as np
from collections import deque

#between node & gate: for node: node.out == None, gate: node.out != None
# if it's a node, get its value or grad directly, otherwise, get gate.out
def getVal(node):
	if node.out != None:
		return node.out.val
	else:
		return node.val

def update(node, val):
	if node.out:
		node.out.val = val
	else:
		node.grad = val

def getGradient(node):
	if node.out:
		return node.out.grad
	else:
		return node.grad

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
		if type(xVal) == int and type(yVal) == int:
			self.out.val = xVal*yVal
		else:
			self.out.val = np.matmul(yVal, xVal)
		return self.out

	def backward(self, z):
		xVal = getVal(self.x)
		yVal = getVal(self.y)
		zVal = getVal(z)
		if type(xVal) == int and type(yVal) == int:
			self.x.grad += zVal*yVal
			self.y.grad += zVal*xVal
		else:
			self.x.grad += np.matmul(xVal.T,dz)
			self.y.grad += np.matmal(yVal.T,dz)

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

	def backward(self, z):
		dz = getVal(z)
		dx = getGradient(self.x)
		dy = getGradient(self.y)
		update(self.x, dx + dz)
		update(self.y, dy + dz)


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
		self.out.val = max(x.val,y.val)
		return self.out

	def backward(self, z):
		dz = getVal(z)
		if (self.x > self.y):
			self.x.grad += dz 
		else:
			self.y.grad += dz

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

	def backward(self, z):
		dz = getVal(z)
		self.out.grad = (self.out.val*(1 - self.out.val))*dz

	def sigmoidFunc(self, value):
		return 1.0/(np.exp(-1*value) + 1)

class softmaxLossGate():
	def __init__(self, x):
		self.label = "softmaxLoss"
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

	def backward(self, z):
		self.x.grad = getVal(z) - getVal(self.x)

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
			return val/(1.0*val)


class Node():
	def __init__(self, val, grad, label=""):
		self.label = label
		self.type = "node"
		self.val = val
		self.out = None
		self.grad = grad
		self.next = None
		self.degree = 0

class Pipeline():
	def __init__(self, adjList):
		self.pipeline = adjList #adjacency list of gates/notes

	def topologicalSort(self):
		count = len(self.pipeline)
		if len(self.pipeline) == 0:
			print "come on it's empty"
			return
		Queue = deque([node for node in self.pipeline if node.degree ==0])
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

				



