# Algorithm Implementations


 Done: 
 1. computational graph implementation inspired by Karpathy's blog post & cs231n: http://karpathy.github.io/neuralnets/
  * basic operations: *, +, sigmoid, max and softmaxCrossEntropyError
  * tological sort for the execution order
  * gradient descent (most basic)
  
 TODO:
 1. Fix vector multiplication & run tests 
 2. (optional improvement) Cost function parser (convert "W*x + b" -> "g1 = multiplyGate(W, x)
                                                g2 = sumGate(g1, b)")
 3. Different versions of SGD (wip): normal SGD, SVRG (Johnson & Zhang, NIPS 2013), Adam (Kingma & Lei Ba, ICLR 2015)
 4. RNN gate, GRU gate
