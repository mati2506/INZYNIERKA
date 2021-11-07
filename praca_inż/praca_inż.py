import numpy as np
import math as mt
import pandas as pd
import copy
import time
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
                if np.sum(np.isnan(merged_weight[j])) == np.size(merged_weight[j]):
                    tmp_ind.append((0,0))
                    tmp_val.append(np.NaN)
                else:
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

    def _outs_of_single_neuron(self, X, weight, bias, number, index):
        outs = []
        for j in range(self.samples_count):
            activation_i = X[j].copy()
            for i in range(number):
                sum_out = np.dot(activation_i, weight[i]) + bias[i]
                activation_i = self._sigmoid(sum_out)
            outs.append(activation_i[index[0]]*weight[number][index])
        return np.array(outs)

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

        for i in range(self.hidden_count+1):
            merged_weight[i][merged_weight[i] == 0] = np.NaN
        
        for i in range(int(zero_weigths), numbers_for_pruning, 1):      
            tmp_ind = []
            tmp_val = []
            for j in range(self.hidden_count+1):
                if np.sum(np.isnan(merged_weight[j])) == np.size(merged_weight[j]):
                    tmp_ind.append((0,0))
                    tmp_val.append(np.NaN)
                else:
                    tmp_ind.append(np.unravel_index(np.nanargmin(np.abs(merged_weight[j])),shape=merged_weight[j].shape))
                    tmp_val.append(merged_weight[j][tmp_ind[j]])
            tmp = np.nanargmin(np.abs(np.array(tmp_val)))

            merged_bias[tmp][tmp_ind[tmp][1]] = merged_bias[tmp][tmp_ind[tmp][1]] + np.mean(self._outs_of_single_neuron(X, weight_for_amendment, bias_for_amendment, tmp, tmp_ind[tmp]))
            merged_weight[tmp][tmp_ind[tmp]] = np.NaN

        for i in range(self.hidden_count+1):
            merged_weight[i][np.isnan(merged_weight[i])] = 0

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

    def pruning_by_variance(self, factor, X): #factor - procentowa liczba połączeń do usunięcia, X - zbiór trenujący
        connect_count = self.feature_count*self.hidden[0]
        for i in range(0,self.hidden_count-1,1):
            connect_count = connect_count + self.hidden[i]*self.hidden[i+1]
        connect_count = connect_count + self.hidden[self.hidden_count-1]*self.class_count
        numbers_for_pruning = int(np.floor(connect_count*factor/100))
        
        merged_weight = copy.deepcopy(self.weight_hidden)
        merged_weight.append(self.weight_out.copy())
        merged_bias = copy.deepcopy(self.bias_hidden)
        merged_bias.append(self.bias_out.copy())
        weight_for_calculation = copy.deepcopy(merged_weight)
        bias_for_calculation = copy.deepcopy(merged_bias)

        variances = []
        means = []
        for i in range(self.hidden_count+1):
            var_tmp1 = []
            mean_tmp1 = []
            for j in range(weight_for_calculation[i].shape[0]):
                var_tmp2 = []
                mean_tmp2 = []
                for k in range(weight_for_calculation[i].shape[1]):
                    outs = self._outs_of_single_neuron(X, weight_for_calculation, bias_for_calculation, i, (j, k))
                    var_tmp2.append(np.var(outs))
                    mean_tmp2.append(np.mean(outs))
                var_tmp1.append(var_tmp2)
                mean_tmp1.append(mean_tmp2)
            variances.append(np.array(var_tmp1))
            means.append(np.array(mean_tmp1))

        for i in range(numbers_for_pruning):      
            tmp_ind = []
            tmp_val = []
            for j in range(self.hidden_count+1):
                if np.sum(np.isnan(variances[j])) == np.size(variances[j]):
                    tmp_ind.append((0,0))
                    tmp_val.append(np.NaN)
                else:
                    tmp_ind.append(np.unravel_index(np.nanargmin(np.abs(variances[j])),shape=variances[j].shape))
                    tmp_val.append(variances[j][tmp_ind[j]])
            tmp = np.nanargmin(np.abs(np.array(tmp_val)))

            merged_bias[tmp][tmp_ind[tmp][1]] = merged_bias[tmp][tmp_ind[tmp][1]] + means[tmp][tmp_ind[tmp]]
            merged_weight[tmp][tmp_ind[tmp]] = 0
            variances[tmp][tmp_ind[tmp]] = np.NaN

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

    def accuracy(self, y_real, y_out):
        count = y_real.shape[0]
        counter = 0
        for i in range(count):
            if_various = 0
            for j in range(y_real.shape[1]):
                if y_real[i][j] != y_out[i][j]:
                    if_various = 1
                    break
            if if_various == 0:
                counter = counter+1
        return (counter*100)/count


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
    
    alpha = 10 #% liczby połączeń do usunięcia przy przycinaniu (w wersji bez pętli)
    name = "test" #prefix nazwy pliku/wykresu do którego będą zapisywane dane

    #mlp1 = my_MLP(hidden=(50),mono=True)
    mlp1 = my_MLP(hidden=(15,10,5), epochs=300)
    mlp1.fit(X_train, y_train)
    
    _, y_pred = mlp1.predict(X_test)

    accuracy_test = mlp1.accuracy(y_test, y_pred)
    print("Dokładność klasyfikacji zbioru testowego: " + str(accuracy_test) + "%")
    print()

    accuracies = []
    times = []
    #if True: #jeżeli ma być bez pętli
    for alpha in range(0,96,1): #pętla po % liczby połączeń do usunięcia przy przycinaniu
        mlp1_cop = mlp1.copy()
        start1 = time.process_time()
        pruning_count = mlp1_cop.simple_pruning(alpha)
        end1 = time.process_time()
        _, y_pred_cop = mlp1_cop.predict(X_test)
        accuracy_test_cop = mlp1_cop.accuracy(y_test, y_pred_cop)

        mlp1_cop2 = mlp1.copy()
        start2 = time.process_time()
        pruning_count2 = mlp1_cop2.simple_pruning_amendment(alpha, X_train)
        end2 = time.process_time()
        _, y_pred_cop2 = mlp1_cop2.predict(X_test)
        accuracy_test_cop2 = mlp1_cop2.accuracy(y_test, y_pred_cop2)       

        mlp1_cop3 = mlp1.copy()
        start3 = time.process_time()
        pruning_count3 = mlp1_cop3.pruning_by_variance(alpha, X_train)
        end3 = time.process_time()        
        _, y_pred_cop3 = mlp1_cop3.predict(X_test)
        accuracy_test_cop3 = mlp1_cop3.accuracy(y_test, y_pred_cop3)


        #print("Dokładność klasyfikacji zbioru testowego po przycinaniu metodą najmniejszych wag: " + str(accuracy_test_cop) + "%")
        #print("Czas trwania przycinania metodą najmniejszych wag: " + str(end1-start1) + "s")
        #print()

        #print("Dokładność klasyfikacji zbioru testowego po przycinaniu metodą najmniejszych wag z poprawką: " + str(accuracy_test_cop2) + "%")
        #print("Czas trwania przycinania metodą najmniejszych wag z poprawką: " + str(end2-start2) + "s")
        #print()

        #print("Dokładność klasyfikacji zbioru testowego po przycinaniu metodą najmniejszych wariancji: " + str(accuracy_test_cop3) + "%")
        #print("Czas trwania przycinania metodą najmniejszych wariancji: " + str(end3-start3) + "s")
        #print()

        #print("Liczba połączeń, które były usuwane: " + str(pruning_count))
        #print()

        accuracies.append([alpha, pruning_count, accuracy_test_cop, accuracy_test_cop2, accuracy_test_cop3])
        times.append([alpha, (end1-start1), (end2-start2), (end3-start3)])


    accuracies = np.array(accuracies)
    accuracies_data = pd.DataFrame(accuracies, columns=["Alpha", "Liczba usuniętych połączeń", "Metoda najmniejszych wag",
                                                        "Metoda najmniejszych wag z poprawką", "Metoda najmniejszych wariancji"])
    times_data = pd.DataFrame(np.array(times), columns=["Alpha", "Metoda najmniejszych wag",
                                                        "Metoda najmniejszych wag z poprawką", "Metoda najmniejszych wariancji"])
    accuracies_data.to_csv(("wyniki/"+name+"_dokładności.csv"), index=False)
    times_data.to_csv(("wyniki/"+name+"_czasy.csv"), index=False)