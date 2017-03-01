#from gates import *
from graph import *
# scalar case: W = 1, x = 2, b =1

W = Node(1,0,"w")
x = Node(2,0, "x")
b = Node(1,0, "b")
y = Node(1,0,"y")
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
g6 = softmaxLossGate(g5)

ppl  = Pipeline([g3, g4, g6, g2, g1,g5])
order =  ppl.topologicalSort()

# each iteration: forward, backprop, optimization
#forward pop
for gate in order:
	gate.forward()
	#print getVal(gate.out), gate.label

#backpop
g5.out.grad = getVal(y) - getVal(g5)
for gate in order[::-1]:
	#the last node
	gate.backward()




#test example from Karpathy's website

a = Node(1, 0, "a")
b = Node(2, 0, "b")
c = Node(-3, 0, "c")
x = Node(-1, 0, "x")
y = Node(3,0, "y")

g1 = multiplyGate(a,x)
g2 = multiplyGate(b,y)
g3 = sumGate(g1, g2)
g4 = sumGate(g3, c)
g5 = sigmoidGate(g4)

weights = [a,b,c,x,y]
ppl  = Pipeline([g3, g4, g2, g1,g5])
order =  ppl.topologicalSort()

#forward prop
for gate in order:
	gate.forward()

print 'circuit output: ', getVal(g5.out) #0.8808 yay

# backprop: start from end
g5.out.grad = 1
for gate in order[::-1]:
	#the last node
	gate.backward()

#gradient descent for each weight vector
step_size = 0.01
for weight in weights:
	weight.val += step_size*weight.grad

for gate in order:
	gate.forward()

print 'circuit output: ', getVal(g5.out) #0.8826 woohoo

