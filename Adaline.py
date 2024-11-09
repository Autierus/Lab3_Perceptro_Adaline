from random import random
from functools import reduce

class Adaline:
    def __init__(self, length, rate, maxEpocas):
        self.weights = [float('%.4f' % random()) for _ in range(0, length + 1)]
        self.epocas = 0
        self.learnRate = rate
        self.maxEpocas = maxEpocas

    def getWeights(self): return self.weights
    def getEpocas(self): return self.epocas
    def atMaxEpocas(self): return self.epocas >= self.maxEpocas

    def train(self, data, expected):
        total_error = 1.0
        while not self.atMaxEpocas() and total_error > 0.001:
            total_error = 0.0
            for (sd, se) in zip(data, expected):
                error_value = self.updateWeight(sd, se)
                total_error += error_value ** 2
            self.epocas += 1
            total_error /= len(data)  # erro médio
        return

    def calculate(self, sData):
        vs = [d * w for (d, w) in zip([-1] + sData, self.weights)]
        return reduce((lambda x, y: x + y), vs) 

    def calculateAll(self, data):
        return [self.activation(self.calculate(d)) for d in data]

    def updateWeight(self, sData, sExpected):
        output = self.calculate(sData)
        error = sExpected - output
        self.weights = [w + self.learnRate * error * x for (w, x) in zip(self.weights, [-1] + sData)]
        return error

    def activation(self, value):
        return 1 if value >= 0 else -1  # converte para -1 ou 1

def parseForTraining(fn):
    i = []
    d = []
    with open(fn) as f:
        lines = f.readlines()
        lines.pop(0)  
        for line in lines:
            args, expected = parseLine(line)
            i.append(args)
            d.append(expected)
    return (i, d)

def parseLine(line):
    w = list(filter(None, line.split(' ')))
    args = []
    for i in range(0, len(w) - 1):
        args.append(float(w[i]))
    return (args, int(float(w[-1])))

def parseInput(fn):
    xs = []
    with open(fn) as f:
        lines = f.readlines()
        lines.pop(0) 
        for line in lines:
            args = list(map(float, filter(None, line.split(' '))))
            xs.append(args)
    return xs

def compare(exp, res):
    equals = [r for (e, r) in zip(exp, res) if e == r]
    return ((len(equals) * 100) / len(res))

# Execução do Adaline
t = Adaline(3, 0.001, 6000)  # ajustada taxa de aprendizado
(datas, desireds) = parseForTraining('res/anexo1.txt')


print('<< Começando Fase de Treinamento >>')
print('Pesos pré-treinamento:', t.getWeights())
t.train(datas, desireds)
print('Treinamento executado')
print('Pesos pós-treinamento:', t.getWeights())
print('Número de Épocas:', t.getEpocas())

nResults = t.calculateAll(datas)
print('Taxa de acerto: {:.02f}%'.format(compare(desireds, nResults)))

print('<< Executando Classificação >>')
datas = parseInput('res/calculo.txt')
nResults = t.calculateAll(datas)
for (i, d) in zip(datas, nResults):
    print('{:+.4f} <- {}'.format(d, i))
