#from gates import *
from graph import *
# scalar case: W = 1, x = 2, b =1

W = Node(1,0,"w")
x = Node(2,0, "x")
b = Node(1,0, "b")
y = Node(1,0,"y")

g1 = multiplyGate(W,x)
g2 = sumGate(g1, b)
#first layer
g3 = sigmoidGate(g2)
#second layer
g4 = sigmoidGate(g3)
#final layer
g5 = softmaxLossGate(g4)

g5.next = y

ppl  = Pipeline([g3, g4, g2, g1,g5])
order =  ppl.topologicalSort()


# each iteration: forward, backprop, optimization
#forward pop
for gate in order:
	gate.forward()
	#print getVal(gate.out), gate.label

#backpop
#dz = y
gradient = 0
for gate in order[::-1]:
	#the last node
	gate.backward()



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

for gate in order:
	gate.forward()

print 'circuit output: ', getVal(g5.out) #0.8808 yay

out = Node(1,0,"out")
g5.out.grad = 1
for gate in order[::-1]:
	#the last node
	gate.backward()

#gradient descent
step_size = 0.01
a.val += step_size*a.grad
print a.grad
b.val += step_size*b.grad
print b.grad
c.val += step_size*c.grad
print c.grad
x.val += step_size*x.grad
print x.grad
y.val += step_size*y.grad
print y.grad
for gate in order:
	gate.forward()

print 'circuit output: ', getVal(g5.out)

