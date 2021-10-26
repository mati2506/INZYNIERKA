import numpy as np
import math as mt
import copy
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class my_MLP(object):
    def __init__(self, hidden=(10, 10, 10), epochs=100, eta=0.1, shuffle=True, mono=False):
        self.hidden = hidden    #Liczba neuronów na kolejnych warstwach ukrytych
        if mono:
            self.hidden_count = 1
            tmp = []
            tmp.append(self.hidden)
            self.hidden = tmp.copy()
        else:
            self.hidden_count = len(hidden) #Liczba powłok ukrytych
        self.epochs = epochs    #Liczba epok
        self.eta = eta          #Współczynnik uczenia
        self.shuffle = shuffle  #Czy mieszać próbki w epokach
        self.one = mono         #czy tylko 1 warstwa ukryta

    def _sigmoid(self, z):
        result = []
        for i in range(len(z)):
            result.append((1.0)/(1.0+mt.exp((-1)*z[i])))
        return np.array(result)

    def _forward(self, X):
        activation_hidden = []
        activation_hidden_i = X.copy()
        for i in range(self.hidden_count):
            sum_out_hidden = np.dot(activation_hidden_i, self.weight_hidden[i]) + self.bias_hidden[i]
            activation_hidden_i = self._sigmoid(sum_out_hidden)
            activation_hidden.append(activation_hidden_i)
        sum_out_out = np.dot(activation_hidden_i, self.weight_out) + self.bias_out
        activation_out = self._sigmoid(sum_out_out)
        return activation_hidden, activation_out

    def fit(self, X, y):
        self.samples_count = X.shape[0] #liczba próbek uczących
        self.feature_count = X.shape[1] #liczba cech
        self.class_count = y.shape[1] #liczba klas
        self.weight_hidden = []
        self.bias_hidden = []
        self.weight_hidden.append(np.random.normal(0,0.1,size=(self.feature_count, self.hidden[0])))
        self.bias_hidden.append(np.zeros(self.hidden[0]))
        for i in range(self.hidden_count-1):
            self.weight_hidden.append(np.random.normal(0,0.1,size=(self.hidden[i], self.hidden[i+1])))
            self.bias_hidden.append(np.zeros(self.hidden[i+1]))
        self.weight_out = np.random.normal(0,0.1,size=(self.hidden[self.hidden_count-1], self.class_count))
        self.bias_out = np.zeros(self.class_count)

        for i in range(self.epochs):
            indexes = np.array(range(self.samples_count))

            if self.shuffle == True:
                indexes = shuffle(indexes)

            for ind in indexes:
                activation_hidden, activation_out = self._forward(X[ind])
                deri_out = activation_out*(1-activation_out)
                delta_out = (activation_out - y[ind])*deri_out
                grad_weight_out = np.outer(activation_hidden[self.hidden_count-1], delta_out)
                grad_bias_out = delta_out
                grad_weight_hidden = []
                grad_bias_hidden = []
                if self.hidden_count > 1:
                    deri_hidden = activation_hidden[self.hidden_count-1]*(1-activation_hidden[self.hidden_count-1])
                    delta_hidden = np.dot(delta_out, np.transpose(self.weight_out))*deri_hidden
                    grad_weight_hidden.append(np.outer(activation_hidden[self.hidden_count-2], delta_hidden))
                    grad_bias_hidden.append(delta_hidden)
                    tmp = delta_hidden.copy()
                    for j in range(self.hidden_count-2,0,-1):
                        deri_hidden = activation_hidden[j]*(1-activation_hidden[j])
                        delta_hidden = np.dot(tmp, np.transpose(self.weight_hidden[j+1]))*deri_hidden
                        grad_weight_hidden.append(np.outer(activation_hidden[j-1], delta_hidden))
                        grad_bias_hidden.append(delta_hidden)
                        tmp = delta_hidden.copy()
                else:
                    tmp = delta_out.copy()
                deri_hidden = activation_hidden[0]*(1-activation_hidden[0])
                if self.one:
                    delta_hidden = np.dot(tmp, np.transpose(self.weight_out))*deri_hidden
                else:
                    delta_hidden = np.dot(tmp, np.transpose(self.weight_hidden[1]))*deri_hidden
                grad_weight_hidden.append(np.outer(X[ind], delta_hidden))
                grad_bias_hidden.append(delta_hidden)

                self.weight_out = self.weight_out - self.eta*grad_weight_out
                self.bias_out = self.bias_out - self.eta*grad_bias_out
                for j in range(self.hidden_count):
                    self.weight_hidden[j] = self.weight_hidden[j] - self.eta*grad_weight_hidden[self.hidden_count-1-j]
                    self.bias_hidden[j] = self.bias_hidden[j] - self.eta*grad_bias_hidden[self.hidden_count-1-j]


    def predict(self, X): #zwraca: [0] - prawdopodobieństwa dopasowania; [1] - dopasowana klasa
        samples_count = X.shape[0] #liczba próbek testujących
        predictions = []
        classes = []

        for i in range(samples_count):
            _, probability = self._forward(X[i])
            predictions.append(probability)          
            index = np.argmax(probability)
            sample = np.zeros(self.class_count)
            sample[index] = 1
            classes.append(sample)       
        return np.array(predictions), np.array(classes)

    def copy(self):
        new_instance = my_MLP(self.hidden, self.epochs, self.eta, self.shuffle)
        new_instance.one = self.one
        new_instance.samples_count = self.samples_count
        new_instance.feature_count = self.feature_count
        new_instance.class_count = self.class_count
        new_instance.weight_hidden = copy.deepcopy(self.weight_hidden)
        new_instance.bias_hidden = copy.deepcopy(self.bias_hidden)
        new_instance.weight_out = copy.deepcopy(self.weight_out)
        new_instance.bias_out = copy.deepcopy(self.bias_out)
        return new_instance

    def simple_pruning(self, factor): #factor - procentowa liczba połączeń do usunięcia
        connect_count = self.feature_count*self.hidden[0]
        for i in range(0,self.hidden_count-1,1):
            connect_count = connect_count + self.hidden[i]*self.hidden[i+1]
        connect_count = connect_count + self.hidden[self.hidden_count-1]*self.class_count
        numbers_for_pruning = int(np.floor(connect_count*factor/100))
        
        merged_weight = copy.deepcopy(self.weight_hidden)
        merged_weight.append(self.weight_out.copy())

        for i in range(numbers_for_pruning):
            tmp_ind = []
            tmp_val = []
            for j in range(self.hidden_count+1):
                tmp_ind.append(np.unravel_index(np.nanargmin(np.abs(merged_weight[j])),shape=merged_weight[j].shape))
                tmp_val.append(merged_weight[j][tmp_ind[j]])
            tmp = np.nanargmin(np.abs(np.array(tmp_val)))
            merged_weight[tmp][tmp_ind[tmp]] = np.NaN

        for i in range(self.hidden_count+1):
            merged_weight[i][np.isnan(merged_weight[i])] = 0

        new_weight_hidden = []
        for i in range(self.hidden_count):
            new_weight_hidden.append(merged_weight[i])
        self.weight_hidden = copy.deepcopy(new_weight_hidden)
        self.weight_out = merged_weight[self.hidden_count].copy()

        return numbers_for_pruning

    def _out_of_single_neuron(self, X, weight, bias, number, index):
        activation_i = X.copy()
        for i in range(number+1):
            sum_out = np.dot(activation_i, weight[i]) + bias[i]
            activation_i = self._sigmoid(sum_out)
        return sum_out[index]

    def simple_pruning_amendment(self, factor, X): #factor - procentowa liczba połączeń do usunięcia, X - zbiór trenujący
        connect_count = self.feature_count*self.hidden[0]
        for i in range(0,self.hidden_count-1,1):
            connect_count = connect_count + self.hidden[i]*self.hidden[i+1]
        connect_count = connect_count + self.hidden[self.hidden_count-1]*self.class_count
        numbers_for_pruning = int(np.floor(connect_count*factor/100))
        
        merged_weight = copy.deepcopy(self.weight_hidden)
        merged_weight.append(self.weight_out.copy())
        merged_bias = copy.deepcopy(self.bias_hidden)
        merged_bias.append(self.bias_out.copy())
        weight_for_amendment = copy.deepcopy(merged_weight)
        bias_for_amendment = copy.deepcopy(merged_bias)

        zero_weigths = 0
        for i in range(self.hidden_count+1):
            zero_weigths = zero_weigths + np.sum(merged_weight[i][merged_weight[i] == 0])
        
        for i in range(int(zero_weigths), numbers_for_pruning, 1):
            for j in range(self.hidden_count+1):
                merged_weight[j][merged_weight[j] == 0] = np.NaN

            tmp_ind = []
            tmp_val = []
            for j in range(self.hidden_count+1):
                tmp_ind.append(np.unravel_index(np.nanargmin(np.abs(merged_weight[j])),shape=merged_weight[j].shape))
                tmp_val.append(merged_weight[j][tmp_ind[j]])
            tmp = np.nanargmin(np.abs(np.array(tmp_val)))
            
            outs = []
            for j in range(self.samples_count):
                outs.append(self._out_of_single_neuron(X[j], weight_for_amendment, bias_for_amendment, tmp, tmp_ind[tmp][1]))

            merged_bias[tmp][1] = merged_bias[tmp][1] + np.mean(np.array(outs))
            merged_weight[tmp][tmp_ind[tmp]] = 0

            for j in range(self.hidden_count+1):
                merged_weight[j][np.isnan(merged_weight[j])] = 0

        new_weight_hidden = []
        new_bias_hidden = []
        for i in range(self.hidden_count):
            new_weight_hidden.append(merged_weight[i])
            new_bias_hidden.append(merged_bias[i])
        self.weight_hidden = copy.deepcopy(new_weight_hidden)
        self.bias_hidden = copy.deepcopy(new_bias_hidden)
        self.weight_out = merged_weight[self.hidden_count].copy()
        self.bias_out = merged_bias[self.hidden_count].copy()

        return numbers_for_pruning

def dokladnosc(y_r, y_w):
    liczba = y_r.shape[0]
    licznik = 0
    for i in range(liczba):
        czy_rozne = 0
        for j in range(y_r.shape[1]):
            if y_r[i][j] != y_w[i][j]:
                czy_rozne = 1
                break
        if czy_rozne == 0:
            licznik = licznik+1
    wynik = (licznik*100)/liczba
    return wynik


if __name__ == '__main__':
    X_iris, y_iris = fetch_openml(name="iris", version=1, return_X_y=True)

    y_iris_coded=[]
    for i in range(len(y_iris)):
        if y_iris[i] == 'Iris-setosa':
            y_iris_coded.append([1.,0.,0.])
        elif y_iris[i] == 'Iris-versicolor':
            y_iris_coded.append([0.,1.,0.])
        else:
            y_iris_coded.append([0.,0.,1.])

    y_iris_coded = np.array(y_iris_coded)

    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris_coded, random_state=13)
    
    #mlp1 = my_MLP(hidden=(15),mono=True)
    mlp1 = my_MLP(hidden=(15,10,5),epochs=300)
    mlp1.fit(X_train, y_train)
    
    _, y_pred = mlp1.predict(X_test)

    dokladnosc_test = dokladnosc(y_test, y_pred)
    print("Dokładność klasyfikacji zbioru testowego:")
    print(dokladnosc_test)
    print()


    mlp1_cop = mlp1.copy()
    pruning_count = mlp1_cop.simple_pruning(15)

    _, y_pred_cop = mlp1_cop.predict(X_test)

    #print(pruning_count)
    dokladnosc_test_cop = dokladnosc(y_test, y_pred_cop)
    print("Dokładność klasyfikacji zbioru testowego po przycinaniu:")
    print(dokladnosc_test_cop)
    print()


    mlp1_cop2 = mlp1.copy()
    pruning_count2 = mlp1_cop2.simple_pruning_amendment(15, X_train)

    _, y_pred_cop2 = mlp1_cop2.predict(X_test)

    #print(pruning_count2)
    dokladnosc_test_cop2 = dokladnosc(y_test, y_pred_cop2)
    print("Dokładność klasyfikacji zbioru testowego po przycinaniu z poprawką:")
    print(dokladnosc_test_cop2)
    print()