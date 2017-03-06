from graph import *
def initializeEmptyVec(x):
	if np.isscalar(x):
		return np.zeros(x)
	elif len(np.shape(x)) == 1:
		return np.empty(x.shape[0])
	else:
		shape = x.shape
		return np.empty([shape[0], shape[1]])

def initializeRandomVec(x):
	if np.isscalar(x):
		return np.random.rand(x)
	elif len(np.shape(x)) == 1:
		return np.random.rand(x.shape[0])
	else:
		shape = x.shape
		return np.random.rand([shape[0], shape[1]])

class Training():
	def __init__(self, inputs):
		self.weights = [i for i in inputs if i.type == "weight"]
		self.gradients = initializeEmptyVec(np.array(inputs))
		self.gates = [i for i in inputs if i.type == "gate" or i.type == "loss"]
		self.pipeline = self.topologicalSort(inputs)

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

	def topologicalSort(self, inputs):
		count = len(self.gates)
		if len(self.gates) == 0:
			print "EMPTY LIST"
			return
		Queue = deque([node for node in inputs if node.degree == 0])
		order = []
		labels = []
		while Queue:
			node = Queue.popleft()
			# we only store the order of gates (operators & loss functions)
			if node.type == "gate" or node.type == "loss":
				order.append(node)
				labels.append(node.label)
			for nex in node.next:
				nex.degree -=1
				if nex.degree ==  0:
					Queue.append(nex)
		print labels
		if len(order) == count:
			print labels
			return order
		else:
			assert False, "not a valid topological order"
			return

