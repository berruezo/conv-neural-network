#coding:utf-8
'''
Implementación de una red neuronal simple
Se entrena con un conjunto de tests mediante backpropagation
'''
import numpy as np
import time

LR = 0.01 #Para gradient descent
MIN_ERROR = .005 #Mínimo error de la red a partir del cual se considerará entrenada
BATCH_SIZE = 10 #Cantidad de tests que examinará la red para modificar los pesos

class Network:
    def __init__(self, shape):
        '''
        shape: Lista que define la topología de la red. Cada elemento representa el tamaño de una capa (la longitud será el número total de capas)
        '''
        self.weights = [np.random.normal(scale=(1/np.sqrt(shape[l])), size=(shape[l], shape[l+1])) for l in range(len(shape))[:-1]] #Pesos de salida de una neurona [capa][neurona_src][neurona_dst]
        self.biases = [abs(np.random.normal(size=(1, n))) for n in shape] #Bias de cada capa [capa][0][peso] (los de la capa 0 no se utilizan)
        
        self.outputs        = [np.zeros((BATCH_SIZE, n)) for n in shape] #Salidas (sin activar) de cada neurona para un conjunto de tests [capa][test][neurona]
        self.activations    = [np.zeros((BATCH_SIZE, n)) for n in shape] #Activaciones de cada neurona para un conjunto de tests [capa][test][neurona]
        self.errors         = [np.zeros((BATCH_SIZE, n)) for n in shape] #Error de la salida (sin activar) de cada neurona para un conjunto de tests [capa][test][neurona]
        
    def calcOutputs(self, inputs):
        '''Calcula las salidas y activaciones de cada neurona de la red
        '''
        self.activations[0] = inputs
        for l in range(len(self.outputs))[1:]: #Forward-pass: Desde la primera capa hasta la última
            self.outputs[l] = np.dot(self.activations[l-1], self.weights[l-1]) + self.biases[l]
            self.activations[l] = relu(self.outputs[l])
        self.activations[-1] = softmax(self.outputs[-1])
            
    def calcErrors(self, targets):
        '''Calcula el error de la salida de cada neurona de la red
        '''
        self.errors[-1] = dsoftmax(self.outputs[-1]) * dloglikelihood(self.activations[-1], targets)
        for l in range(len(self.outputs))[-2::-2]: #Backpropagation: Desde la última capa hasta la segunda (el error de la primera no es necesario)
            self.errors[l] = np.dot(self.errors[l+1], self.weights[l].T) * drelu(self.outputs[l])
        
    def train(self, trainInputs, trainOutputs, testInputs, testOutputs):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        trainInputs: Lista de entradas para entrenar la red [test][neurona]
        trainOutpus: Lista de salidas esperadas para entrenar la red [test][neurona]
        testInputs: Lista de entradas para evaluar la red [test][neurona]
        testOutputs: Lista de salidas esperadas para evaluar la red [test][neurona]
        '''
        epochs = 0
        cost = 1.0 #Valor de la función de coste para los pesos y bias actuales
        n_batches = len(trainInputs)/BATCH_SIZE
        
        while cost > MIN_ERROR:
            t = time.time()
            for inputs, targets in zip(np.split(trainInputs, n_batches), np.split(trainOutputs, n_batches)):
                self.calcOutputs(inputs) #Se calcula la salida de cada neurona
                self.calcErrors(targets) #Se calculan los errores de cada neurona
                
                #Se actualizan los pesos y bias en función de los errores calculados
                for l in range(len(self.weights)):
                    self.weights[l] -= np.dot(self.activations[l].T, self.errors[l+1]) * LR
                    self.biases[l+1] -= np.sum(self.errors[l+1], axis=0, keepdims=True) * LR
                
            epochs += 1
            print "Iter:\t%d\nTiempo:\t%f" % (epochs, time.time()-t)
            cost, accuracy = self.test(testInputs, testOutputs)
            print "Cost:\t%f\nAcc:\t%.2f%%\n\n" % (cost, accuracy*100)

    def test(self, testInputs, testOutputs):
        '''Evalúa la red con una lista de tests y devuelve el coste y el porcentaje de aciertos
        '''
        self.calcOutputs(testInputs)
        cost = quadratic(self.activations[-1], testOutputs)
        accuracy = np.sum(np.equal(np.argmax(self.activations[-1],axis=1), np.argmax(testOutputs, axis=1))) / float(len(testInputs))
        return cost, accuracy
        
    def getOutputs(self): return self.activations[-1]
    
#FUNCIONES DE ACTIVACIÓN
#z: Vector de salidas sin activar de una capa (o matriz de la forma [test, neurona])
#return: Vector de salidas activadas
#return (derivative): Vector de derivadas de la salida activada respecto a la salida sin activar
def sigmoid(z): return 1.0/(1.0+np.exp(-z)) #(0, 1)
def dsigmoid(z):
    tmp = sigmoid(z)
    return tmp*(1-tmp)
    
def relu(z): return np.maximum(0, z) #[0, inf) Inicializar los bias a valores positivos
def drelu(z): return np.sign(z)/2 + 0.5

def tanh(z): return np.tanh(z) #(-1, 1)
def dtanh(z): return np.sech(z)**2

def softmax(z): return np.exp(z)/(np.sum(np.exp(z), axis=1)[:,None]) #(0, 1] La suma es 1
def dsoftmax(z):
    tmp = softmax(z)
    return tmp*(1-tmp)

#FUNCIONES DE COSTE
#a: Vector de activaciones de la última capa (o matriz de la forma [test, neurona])
#y: Vector de salidas esperadas de la última capa
#return: Coste (escalar) de toda la red para el test dado
#return (derivative): Vector de derivadas de la función de coste respecto a la activación de cada neurona
def quadratic(a, y): return np.sum(0.5*(a-y)**2) / len(a)
def dquadratic(a, y): return a-y

def loglikelihood(a, y): return np.sum(-np.log(a[range(len(a)), np.argmax(y, axis=1)])) / len(a) #Usar con softmax
def dloglikelihood(a, y): return a-y
#def dloglikelihood(a, y): return (a-y)/(a*(1-a)) / len(a)
