#from gates import *
from graph import *
# scalar case: W = 1, x = 2, b =1

W = Node(1,0,"w")
x = Node(2,0, "x")
b = Node(1,0, "b")
y = Node(3,0,"y")

g1 = multiplyGate(W,x)
g2 = sumGate(g1, b)
g3 = sigmoidGate(g2)

g4 = sigmoidGate(g3)

g5 = softmaxLossGate(g4)

g5.next = y

ppl  = Pipeline([g3, g4, g2, g1,g5])
order =  ppl.topologicalSort()

#forward pop
for gate in order:
	gate.forward()
	#print getVal(gate.out), gate.label

#backpop
#dz = y
gradient = 0
for gate in order[::-1]:
	#the last node
	gate.backward(gate.next)
	print gate.label
	print gate.out.val
