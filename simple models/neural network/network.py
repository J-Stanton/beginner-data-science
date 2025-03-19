import numpy as np

def sumOfSquaresLoss(y,t):
    error = 0
    for k in range(len(y)):
        error += (y[k]-t[k])**2
    return round(0.5*error,4)

def sigmoidActivation(x):
    return 1/(1+np.exp(-x))

def feedForwardStep(a0,W1,W2,b1,b2):
    a1 = sigmoidActivation(np.matmul(a0,W1) + b1)
    a2 = sigmoidActivation(np.matmul(a1,W2)+b2)
    return a1,a2

def backpropagation(a0,a1,a2,W1,W2,b1,b2,t):
    learningRate = 0.1

    deltaOut = (a2 - t)*a2*(1-a2)

    #Calculate hidden layers delta
    deltaHidden = np.zeros(len(a1))
    for k in range(len(a1)):
        sumTerm = 0
        for i in range(len(deltaOut)):
            sumTerm += W2[k, i] * deltaOut[i]
        deltaHidden[k] = sumTerm * a1[k] * (1 - a1[k])

    #Update W2 and b2
    for k in range(len(a1)):
        for i in range(len(deltaOut)):
            W2[k,i] -= learningRate * a1[k] * deltaOut[i]
    b2 -= learningRate * deltaOut

    #Update W1 and b1
    for k in range(len(a0)):
        for i in range(len(deltaHidden)):
            W1[k,i] -= learningRate * a0[k] * deltaHidden[i]
    b1 -= learningRate * deltaHidden

    return W1,b1,W2,b2


def main():
    a0 = np.zeros(5)
    t = np.zeros(3)

    #Get input
    for k in range(len(a0)):
        a0[k] = float(input())
    for k in range(len(t)):
        t[k] = float(input())

    W1 = np.ones((5,10))
    b1 = np.ones(10)

    W2 = np.ones((10, 3))
    b2 = np.ones(3)

    a1,a2 = feedForwardStep(a0,W1,W2,b1,b2)
    print(sumOfSquaresLoss(a2,t))

    W1,b1,W2,b2 = backpropagation(a0,a1,a2,W1,W2,b1,b2,t)
    a1,a2 = feedForwardStep(a0,W1,W2,b1,b2)
    print(sumOfSquaresLoss(a2,t))

if __name__ == "__main__":
    main()

# Test Data
# a0 = np.array([3,1,-2,-1,-4])
# t = np.array([0,1,0])
# a0 = np.array([-3,1,-5,0,-1])
# t = np.array([0.3,0.2,0.7])

# Test Data with input layer 3 layers, hidden 8 layers and output layer 3 layers
# a0 = np.zeros(4)
# t = np.zeros(3)
# W1 = np.ones((4,8))
# b1 = np.ones(8)
# W2 = np.ones((8,3))
# b2 = np.ones(3)