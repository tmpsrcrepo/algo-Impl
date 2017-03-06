from graph import *
# scalar case: W = 1, x = 2, b =1

W = Node(1,0,"w")
x = Node(2,0, "x")
b = Node(1,0, "b")

zero = Node(0,0,"zero")

g1 = multiplyGate(W,x)
g2 = sumGate(g1, b)
#first layer: relu
g3 = maxGate(g2, zero)
#second layer
g4 = sigmoidGate(g3)
#third layer
g5 = sigmoidGate(g4)
#final layer
g6 = softmaxCEGate(g5)

training = Training(gateList = [g3, g4, g6, g2, g1,g5],
	weights = [W,x,b])

# each iteration: forward, backprop, optimization
#forward pop
y = 1
training.forwardTraining(y)
training.backwardTraining(y)
training.gradientDescent(0.01)
