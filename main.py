#coding:utf-8
'''
Clasificador de dígitos escritos a mano
Utiliza la librería Network.py y el dataset de MNIST
'''
import Network as net
import numpy as np

TRAINS = 60000 #Cantidad de tests para entrenar la red
TESTS = 10000 #Cantidad de tests para evaluar la red
SIZE = 784

imgTrain = open("mnist/train-images-idx3-ubyte", "rb")
imgTrain.read(16) #Offset
lblTrain = open("mnist/train-labels-idx1-ubyte", "rb")
lblTrain.read(8)
imgTest = open("mnist/t10k-images-idx3-ubyte", "rb")
imgTest.read(16)
lblTest = open("mnist/t10k-labels-idx1-ubyte", "rb")
lblTest.read(8)

print "Leyendo imágenes de entrenamiento"
trainInputs = np.split(np.fromfile(imgTrain, "uint8", TRAINS*SIZE)/255.0, TRAINS)
trainInputs = np.array(trainInputs) #Esto es lento. Estaría guay quitarlo
trainOutputs = np.zeros((TRAINS, 10))
trainOutputs[range(TRAINS), np.fromfile(lblTrain, "uint8", TRAINS)] = 1

print "Leyendo imágenes de test"
testInputs = np.split(np.fromfile(imgTest, "uint8", TESTS*SIZE)/255.0, TESTS)
testInputs = np.array(testInputs)
testOutputs = np.zeros((TESTS, 10))
testOutputs[range(TESTS), np.fromfile(lblTest, "uint8", TESTS)] = 1

print "Entrenando"
net = net.Network([784,30,10])
net.train(trainInputs, trainOutputs, testInputs, testOutputs)
print "Entrenada"


'''
train = []
test = []

print "Leyendo imagenes de entrenamiento"
for _ in range(60000):
    img = []
    for _ in range(784):
        img.append(ord(imgTrain.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(lblTrain.read(1))] = 1

    train.append([img,lbl])
train = np.array(train)

print "Leyendo imagenes de test"
for _ in range(10000):
    img = []
    for _ in range(784):
        img.append(ord(imgTest.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(lblTest.read(1))] = 1

    test.append([img,lbl])
test = np.array(test)


errores = [0,0,0,0,0,0,0,0,0,0]
for inputs, targets in test:
    net.calcOutputs(inputs)
    target = np.argmax(targets)
    output = np.argmax(net.getOutputs())
    if target != output:
        img = Image.new("P", (28,28))
        img.putdata([i*255 for i in inputs])
        img.save('errores/' + str(np.around(net.getOutputs(), 1)) + ".png")
        errores[output] += 1
'''
