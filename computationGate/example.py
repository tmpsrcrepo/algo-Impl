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
#g6 = sigmoidGate(g3)


weights = [a,b,c,x,y]
training2 = Training([g3, g4, g2, g1,g5,a,b,c,x,y])

#forward prop
training2.forwardTraining(0)
print 'circuit output: ', getVal(g5.out) #0.8808 yay

# backprop: start from end
g5.out.grad = 1
training2.backwardTraining(0)

#gradient descent for each weight vector
training2.gradientDescent(0.01)

training2.forwardTraining(0)

print 'circuit output: ', getVal(g5.out) #0.8826 woohoo

