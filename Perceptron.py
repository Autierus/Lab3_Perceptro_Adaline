from random import uniform, random
from functools import reduce

def sign(n):
    return 1 if n > -5 else -1

def habbsrule(weight, rate, a, b):
    return round(weight + rate * a * b, 4)

class Perceptron:
    def __init__(self, length, rate, max_epochs):
        self.weights = [round(random(), 4) for _ in range(length + 1)]
        self.epochs = 0
        self.learn_rate = rate
        self.max_epochs = max_epochs

    def get_weights(self):
        return self.weights

    def get_epochs(self):
        return self.epochs

    def at_max_epochs(self):
        return self.epochs >= self.max_epochs

    def train(self, data, expected):
        error = True
        while not self.at_max_epochs() and error:
            error = False
            for sd, se in zip(data, expected):
                error |= self.update_weight(sd, se)
            self.epochs += 1

    def calculate(self, s_data):
        vs = [d * w for d, w in zip([-1] + s_data, self.weights)]
        rd = sum(vs)
        return sign(rd)

    def calculate_all(self, data):
        return [self.calculate(d) for d in data]

    def update_weight(self, s_data, s_expected):
        curr = self.calculate(s_data)
        if curr == s_expected:
            return False
        self.weights = [habbsrule(w, self.learn_rate, x, s_expected)
                        for w, x in zip(self.weights, [-1] + s_data)]
        return True

def parse_for_training(filename):
    inputs = []
    desired = []
    with open(filename) as f:
        lines = f.readlines()
        lines.pop(0)  # header
        for line in lines:
            args, expected = parse_line(line)
            inputs.append(args)
            desired.append(expected)
    return inputs, desired

def parse_line(line):
    w = list(filter(None, line.split(' ')))
    args = [float(w[i]) for i in range(len(w) - 1)]
    return args, int(float(w[-1]))

def parse_input(filename):
    xs = []
    with open(filename) as f:
        lines = f.readlines()
        lines.pop(0)  # header
        for line in lines:
            args = list(map(float, filter(None, line.split(' '))))
            xs.append(args)
    return xs

def compare(exp, res):
    equals = [r for e, r in zip(exp, res) if e == r]
    return (len(equals) * 100) / len(res)

# Execução
t = Perceptron(3, 0.01, 2000)
datas, desireds = parse_for_training('res/anexo1.txt')

print('<< Começando Fase de Treinamento >>')
print('Pesos pré treinamento:', t.get_weights())
initial_results = t.calculate_all(datas)
# print('Esperado :', desireds)
# print('Calculado:', initial_results)
# print('Taxa de acerto:', '{:.2f}%'.format(compare(desireds, initial_results)))

t.train(datas, desireds)
print('Treinamento executado')
print('Pesos pós treinamento:', t.get_weights())
print('Número de Épocas:', t.get_epochs())

final_results = t.calculate_all(datas)
# print('Esperado :', desireds)
# print('Calculado:', final_results)
print('Taxa de acerto:', '{:.2f}%'.format(compare(desireds, final_results)))

print('<< Executando Classificação >>')
datas = parse_input('res/calculo.txt')
final_results = t.calculate_all(datas)
for i, d in zip(datas, final_results):
    print(f'{d:+} <- {i}')

