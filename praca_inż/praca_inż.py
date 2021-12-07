import numpy as np
import math as mt
import pandas as pd
import copy
import time
import matplotlib.pyplot as plt
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

    def fit(self, X, y): #samo uczenie
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

            if self.accuracy(y, self.predict(X)[1]) == 100:
                break

    
    def fit_for_pruning(self, X, y): #uczenie + wyliczanie zmiennej decyzyjnej przycinania metodą Karnin'a
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

        zero_weight_hidden = copy.deepcopy(self.weight_hidden)
        zero_weight_out = self.weight_out.copy()

        s = []
        for i in range(self.hidden_count):
            s.append(np.zeros(self.weight_hidden[i].shape))
        s.append(np.zeros(self.weight_out.shape))

        for i in range(self.epochs):            
            last_weight_hidden = copy.deepcopy(self.weight_hidden)
            last_weight_out = self.weight_out.copy()

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

            s_change = []
            for ii in range(self.hidden_count):
                tmp = np.zeros(self.weight_hidden[ii].shape)
                for ij in range(self.weight_hidden[ii].shape[0]):
                    for ik in range(self.weight_hidden[ii].shape[1]):
                        if self.weight_hidden[ii][ij,ik] != zero_weight_hidden[ii][ij,ik]:
                            tmp[ij,ik] = ((self.weight_hidden[ii][ij,ik]-last_weight_hidden[ii][ij,ik])**2)*self.weight_hidden[ii][ij,ik]/(self.eta*(self.weight_hidden[ii][ij,ik]-zero_weight_hidden[ii][ij,ik]))
                s_change.append(tmp)
            tmp = np.zeros(self.weight_out.shape)
            for ij in range(self.weight_out.shape[0]):
                for ik in range(self.weight_out.shape[1]):
                    tmp[ij,ik] = ((self.weight_out[ij,ik]-last_weight_out[ij,ik])**2)*self.weight_out[ij,ik]/(self.eta*(self.weight_out[ij,ik]-zero_weight_out[ij,ik]))
            s_change.append(tmp)

            for ii in range(self.hidden_count+1):
                s[ii] = s[ii] + s_change[ii]

            if self.accuracy(y, self.predict(X)[1]) == 100:
                break

        return s

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
        if factor <= 0 or factor > 100:
            return 0

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
        if factor <= 0 or factor > 100:
            return 0

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
        if factor <= 0 or factor > 100:
            return 0

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

    def fit_pruning(self, s_in, factor, X):
        if factor <= 0 or factor > 100:
            return 0

        connect_count = self.feature_count*self.hidden[0]
        for i in range(0,self.hidden_count-1,1):
            connect_count = connect_count + self.hidden[i]*self.hidden[i+1]
        connect_count = connect_count + self.hidden[self.hidden_count-1]*self.class_count
        numbers_for_pruning = int(np.floor(connect_count*factor/100))
        
        s = copy.deepcopy(s_in)
        merged_weight = copy.deepcopy(self.weight_hidden)
        merged_weight.append(self.weight_out.copy())
        merged_bias = copy.deepcopy(self.bias_hidden)
        merged_bias.append(self.bias_out.copy())
        weight_for_amendment = copy.deepcopy(merged_weight)
        bias_for_amendment = copy.deepcopy(merged_bias)

        for i in range(numbers_for_pruning):      
            tmp_ind = []
            tmp_val = []
            for j in range(self.hidden_count+1):
                if np.sum(np.isnan(s[j])) == np.size(s[j]):
                    tmp_ind.append((0,0))
                    tmp_val.append(np.NaN)
                else:
                    tmp_ind.append(np.unravel_index(np.nanargmin(np.abs(s[j])),shape=s[j].shape))
                    tmp_val.append(s[j][tmp_ind[j]])
            tmp = np.nanargmin(np.abs(np.array(tmp_val)))

            merged_bias[tmp][tmp_ind[tmp][1]] = merged_bias[tmp][tmp_ind[tmp][1]] + np.mean(self._outs_of_single_neuron(X, weight_for_amendment, bias_for_amendment, tmp, tmp_ind[tmp]))
            merged_weight[tmp][tmp_ind[tmp]] = 0
            s[tmp][tmp_ind[tmp]] = np.NaN

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
            if np.argmax(y_real[i]) == np.argmax(y_out[i]):
                counter = counter+1
        return (counter*100)/count


if __name__ == '__main__':
    #USTAWIENIA TESTÓW (+ ZMIANY KOMENTARZY W SEKCJI UCZENIA ORAZ SEKCJI PRZYCINANIA)
    alpha = 40 #% liczby połączeń do usunięcia przy przycinaniu (w wersji bez pętli)
    which_data = 3 #wybór zbioru do wczytania

    #WCZYTANIE WYBRANYCH DANYCH DO TESTOWANIA
    if which_data == 0:
        name = "test" #prefix nazwy pliku/wykresu do którego będą zapisywane dane
        data = pd.read_csv('zbiory/iris.data')
        X_iris = data.iloc[:,0:4].to_numpy()
        y_iris=[]
        for i in range(len(X_iris)):
            if data.iloc[i,4] == 'Iris-setosa':
                y_iris.append([1.,0.,0.])
            elif data.iloc[i,4] == 'Iris-versicolor':
                y_iris.append([0.,1.,0.])
            else:
                y_iris.append([0.,0.,1.])
        y_iris = np.array(y_iris)

        X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=2)

    elif which_data == 1: #ok. 5-7h dokładość test ok. 49,6% (stałe - dany podział train i test)
        name = "first-order" #prefix nazwy pliku/wykresu do którego będą zapisywane dane

        data_train = pd.read_csv('zbiory/train_first-order.csv',header=None)
        train = data_train.to_numpy()
        X_train = train[:,0:51]
        y_train = train[:,51:]
        y_train[y_train==(-1)] = 0
        data_test = pd.read_csv('zbiory/test_first-order.csv',header=None)
        test = data_test.to_numpy()
        X_test = test[:,0:51]
        y_test = test[:,51:]
        y_test[y_test==(-1)] = 0

    elif which_data == 2: #ok. 14-16h dokładność test: ok. 90% (losowe - train i test dzielone w kodzie)
        name = "Dry_Bean" #prefix nazwy pliku/wykresu do którego będą zapisywane dane

        data = pd.read_excel('zbiory/Dry_Bean_Dataset.xlsx')
        X_bean = data.iloc[:,0:16].to_numpy()
        X_bean[:,0] = X_bean[:,0]/10000
        X_bean[:,1] = X_bean[:,1]/100
        X_bean[:,2] = X_bean[:,2]/100
        X_bean[:,3] = X_bean[:,3]/100
        X_bean[:,6] = X_bean[:,6]/10000
        X_bean[:,7] = X_bean[:,7]/100
        y_bean = []
        for i in range(X_bean.shape[0]):
            if data.iloc[i,16] == 'SEKER':
                y_bean.append([1.,0.,0.,0.,0.,0.,0.])
            elif data.iloc[i,16] == 'BARBUNYA':
                y_bean.append([0.,1.,0.,0.,0.,0.,0.])
            elif data.iloc[i,16] == 'BOMBAY':
                y_bean.append([0.,0.,1.,0.,0.,0.,0.])
            elif data.iloc[i,16] == 'CALI':
                y_bean.append([0.,0.,0.,1.,0.,0.,0.])
            elif data.iloc[i,16] == 'HOROZ':
                y_bean.append([0.,0.,0.,0.,1.,0.,0.])
            elif data.iloc[i,16] == 'SIRA':
                y_bean.append([0.,0.,0.,0.,0.,1.,0.])
            elif data.iloc[i,16] == 'DERMASON':
                y_bean.append([0.,0.,0.,0.,0.,0.,1.])
        y_bean = np.array(y_bean)

        X_train, X_test, y_train, y_test = train_test_split(X_bean, y_bean, random_state=2, test_size=0.4)

    elif which_data == 3: #ok. 19-21h dokładność test: ok. 49.33% (stałe - dany podział train i test)
        name = "Crowdsourced_Mapping" #prefix nazwy pliku/wykresu do którego będą zapisywane dane

        data_train = pd.read_csv('zbiory/training_Crowdsourced.csv')
        X_train = data_train.iloc[:,1:].to_numpy()
        X_train = X_train/1000
        y_train = []
        for i in range(X_train.shape[0]):
            if data_train.iloc[i,0] == 'impervious':
                y_train.append([1.,0.,0.,0.,0.,0.])
            elif data_train.iloc[i,0] == 'farm':
                y_train.append([0.,1.,0.,0.,0.,0.])
            elif data_train.iloc[i,0] == 'forest':
                y_train.append([0.,0.,1.,0.,0.,0.])
            elif data_train.iloc[i,0] == 'grass':
                y_train.append([0.,0.,0.,1.,0.,0.])
            elif data_train.iloc[i,0] == 'orchard':
                y_train.append([0.,0.,0.,0.,1.,0.])
            elif data_train.iloc[i,0] == 'water':
                y_train.append([0.,0.,0.,0.,0.,1.])
        y_train = np.array(y_train)
        data_test = pd.read_csv('zbiory/testing_Crowdsourced.csv')
        X_test = data_test.iloc[:,1:].to_numpy()
        X_test = X_test/1000
        y_test = []
        for i in range(X_test.shape[0]):
            if data_test.iloc[i,0] == 'impervious':
                y_test.append([1.,0.,0.,0.,0.,0.])
            elif data_test.iloc[i,0] == 'farm':
                y_test.append([0.,1.,0.,0.,0.,0.])
            elif data_test.iloc[i,0] == 'forest':
                y_test.append([0.,0.,1.,0.,0.,0.])
            elif data_test.iloc[i,0] == 'grass':
                y_test.append([0.,0.,0.,1.,0.,0.])
            elif data_test.iloc[i,0] == 'orchard':
                y_test.append([0.,0.,0.,0.,1.,0.])
            elif data_test.iloc[i,0] == 'water':
                y_test.append([0.,0.,0.,0.,0.,1.])
        y_test = np.array(y_test)

    elif which_data == 4: #zbiór chyba do niczego
        name = "Wilt" #prefix nazwy pliku/wykresu do którego będą zapisywane dane

        data_train = pd.read_csv('zbiory/training_Wilt.csv')
        X_train = data_train.iloc[:,1:].to_numpy()
        X_train = X_train/100
        y_train = []
        for i in range(X_train.shape[0]):
            if data_train.iloc[i,0] == 'w':
                y_train.append([1.,0.])
            elif data_train.iloc[i,0] == 'n':
                y_train.append([0.,1.])
        y_train = np.array(y_train)
        data_test = pd.read_csv('zbiory/testing_Wilt.csv')
        X_test = data_test.iloc[:,1:].to_numpy()
        X_test = X_test/100
        y_test = []
        for i in range(X_test.shape[0]):
            if data_test.iloc[i,0] == 'w':
                y_test.append([1.,0.])
            elif data_test.iloc[i,0] == 'n':
                y_test.append([0.,1.])
        y_test = np.array(y_test)

    elif which_data == 5: #ok. 8-10h dokładność test: ok. 90% (losowe - train i test dzielone w kodzie)
        name = "Electrical_grid" #prefix nazwy pliku/wykresu do którego będą zapisywane dane

        data = pd.read_csv('zbiory/Electrical_Grid_Stability.csv')
        X_grid = data.iloc[:,0:12].to_numpy() #kolumna [12] służy do regresji
        y_grid = []
        for i in range(X_grid.shape[0]):
            if data.iloc[i,13] == 'stable':
                y_grid.append([1.,0.])
            else:
                y_grid.append([0.,1.])
        y_grid = np.array(y_grid)
        
        X_train, X_test, y_train, y_test = train_test_split(X_grid, y_grid, random_state=2, test_size=0.4)
  

    #WYBÓR ARCHITEKTURY SIECI
    #arch = [(10,8,6),(12,9,6),(15,11,7),(17,12,7),(20,14,8)]
    #arch_tx = ["(10,8,6)","(12,9,6)","(15,11,7)","(17,12,7)","(20,14,8)"]

    #pred_tmp = []
    #for i in range(len(arch)):
    #    tmp_mlp = my_MLP(hidden=arch[i], epochs=300)
    #    tmp_mlp.fit(X_train, y_train)
    #    _, y_pred_tmp = tmp_mlp.predict(X_test)
    #    tmp_acc_test = tmp_mlp.accuracy(y_test, y_pred_tmp)
    #    pred_tmp.append(tmp_acc_test)
    #    print("Dokładność klasyfikacji " + arch_tx[i] + ": " + str(tmp_acc_test))
    #pred_tmp_data = pd.DataFrame(np.round(np.array(pred_tmp), 4), index=arch_tx, columns=[name])
    #pred_tmp_data.to_csv(("wyniki/"+name+"_dokładności_architektury.csv"))


    #INICJALIZACJA, UCZENIE I TESTOWANIE SIECI
    #mlp1 = my_MLP(hidden=(50),mono=True)
    mlp1 = my_MLP(hidden=(17,12,7), epochs=300)
    print("Uczenie...")
    start = time.process_time()
    #mlp1.fit(X_train, y_train)
    s = mlp1.fit_for_pruning(X_train, y_train)
    stop = time.process_time()
    print("Uczenie zakończone")
    print("Czas trwania uczenia: " + str(stop-start) + "s")
    print()

    _, y_pred = mlp1.predict(X_test)

    accuracy_test = mlp1.accuracy(y_test, y_pred)
    print("Dokładność klasyfikacji zbioru testowego: " + str(accuracy_test) + "%")
    print()

    #PRZYCINANIE SIECI I TESTOWANIE
    print("Przycinanie...")
    accuracies = []
    times = []
    predict_times = []
    #if True: #jeżeli ma być bez pętli
    for alpha in range(0,101,1): #pętla po % liczby połączeń do usunięcia przy przycinaniu
        print("Aktualna alpha: " + str(alpha))

        mlp1_cop = mlp1.copy()
        start1 = time.process_time()
        pruning_count = mlp1_cop.simple_pruning(alpha)
        end1 = time.process_time()
        start11 = time.process_time()
        _, y_pred_cop = mlp1_cop.predict(X_test)
        end11 = time.process_time()
        accuracy_test_cop = mlp1_cop.accuracy(y_test, y_pred_cop)

        mlp1_cop2 = mlp1.copy()
        start2 = time.process_time()
        pruning_count2 = mlp1_cop2.simple_pruning_amendment(alpha, X_train)
        end2 = time.process_time()
        start21 = time.process_time()
        _, y_pred_cop2 = mlp1_cop2.predict(X_test)
        end21 = time.process_time()
        accuracy_test_cop2 = mlp1_cop2.accuracy(y_test, y_pred_cop2)       

        mlp1_cop3 = mlp1.copy()
        start3 = time.process_time()
        pruning_count3 = mlp1_cop3.pruning_by_variance(alpha, X_train)
        end3 = time.process_time() 
        start31 = time.process_time()
        _, y_pred_cop3 = mlp1_cop3.predict(X_test)
        end31 = time.process_time()
        accuracy_test_cop3 = mlp1_cop3.accuracy(y_test, y_pred_cop3)

        mlp1_cop4 = mlp1.copy()
        start4 = time.process_time()
        pruning_count4 = mlp1_cop4.fit_pruning(s, alpha, X_train)
        end4 = time.process_time()    
        start41 = time.process_time()
        _, y_pred_cop4 = mlp1_cop4.predict(X_test)
        end41 = time.process_time()
        accuracy_test_cop4 = mlp1_cop4.accuracy(y_test, y_pred_cop4)


        #print("Dokładność klasyfikacji zbioru testowego po przycinaniu metodą intuicyjną (najmniejszych wag): " + str(accuracy_test_cop) + "%")
        #print("Czas trwania przycinania metodą intuicyjną (najmniejszych wag): " + str(end1-start1) + "s")
        #print()

        #print("Dokładność klasyfikacji zbioru testowego po przycinaniu ulepszoną metodą intuicyjną (najmniejszych wag z poprawką): " + str(accuracy_test_cop2) + "%")
        #print("Czas trwania przycinania ulepszoną metodą intuicyjną (najmniejszych wag z poprawką): " + str(end2-start2) + "s")
        #print()

        #print("Dokładność klasyfikacji zbioru testowego po przycinaniu metodą najmniejszych wariancji: " + str(accuracy_test_cop3) + "%")
        #print("Czas trwania przycinania metodą najmniejszych wariancji: " + str(end3-start3) + "s")
        #print()

        #print("Dokładność klasyfikacji zbioru testowego po przycinaniu metodą Karnin'a (najmniejszej zmienności wag): " + str(accuracy_test_cop4) + "%")
        #print("Czas trwania przycinania metodą Karnin'a (najmniejszej zmienności wag): " + str(end4-start4) + "s")
        #print()

        #print("Liczba połączeń, które były usuwane: " + str(pruning_count))
        #print()

        accuracies.append([alpha, pruning_count, accuracy_test_cop, accuracy_test_cop2, accuracy_test_cop3, accuracy_test_cop4])
        times.append([alpha, (end1-start1), (end2-start2), (end3-start3), (end4-start4)])
        predict_times.append([alpha, (end11-start11), (end21-start21), (end31-start31), (end41-start41)])


    print()
    print("Generowanie plików wynikowych...")
    accuracies = np.round(np.array(accuracies), 4)
    times = np.round(np.array(times), 4)
    predict_times = np.round(np.array(predict_times), 4)

    #generowanie csv 
    accuracies_data = pd.DataFrame(accuracies, columns=["Alpha", "Liczba usuniętych połączeń", "Metoda intuicyjna",
                                                        "Ulepszona metoda intuicyjna", "Metoda najmniejszych wariancji",
                                                        "Metoda Karnin'a"])
    times_data = pd.DataFrame(times, columns=["Alpha", "Metoda intuicyjna", "Ulepszona metoda intuicyjna",
                                                        "Metoda najmniejszych wariancji", "Metoda Karnin'a"])
    predict_times_data = pd.DataFrame(predict_times, columns=["Alpha", "Metoda intuicyjna", "Ulepszona metoda intuicyjna",
                                                              "Metoda najmniejszych wariancji", "Metoda Karnin'a"])
    accuracies_data.to_csv(("wyniki/"+name+"_dokładności.csv"), index=False)
    times_data.to_csv(("wyniki/"+name+"_czasy.csv"), index=False)
    predict_times_data.to_csv(("wyniki/"+name+"_czasy_klasyfikacji.csv"), index=False)

    #generowanie wykresów
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(accuracies[:,0],accuracies[:,2],label="Metoda intuicyjna")
    ax.plot(accuracies[:,0],accuracies[:,3],label="Ulepszona metoda intuicyjna")
    ax.plot(accuracies[:,0],accuracies[:,4],label="Metoda najmniejszych wariancji")
    ax.plot(accuracies[:,0],accuracies[:,5],label="Metoda Karnin'a")
    plt.title("Dokładności klasyfikacji dla zbioru " + name)
    plt.xlabel("Procent usuniętych połączeń")
    plt.ylabel("Dokładność klasyfikacji zbioru testowego")
    plt.xlim(0, 100)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + pos.height*0.25, pos.width, pos.height*0.75])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
    plt.savefig("wyniki/"+name+"_dokładności.png")
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(times[:,0],times[:,1],label="Metoda intuicyjna")
    ax.plot(times[:,0],times[:,2],label="Ulepszona metoda intuicyjna")
    ax.plot(times[:,0],times[:,3],label="Metoda najmniejszych wariancji")
    ax.plot(times[:,0],times[:,4],label="Metoda Karnin'a")
    plt.title("Czasy trwania przycinania dla zbioru " + name)
    plt.xlabel("Procent usuniętych połączeń")
    plt.ylabel("Czas przycinania [s]")
    plt.xlim(0, 100)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + pos.height * 0.25, pos.width, pos.height * 0.75])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
    plt.savefig("wyniki/"+name+"_czasy.png")

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(predict_times[:,0],predict_times[:,1],label="Metoda intuicyjna")
    ax.plot(predict_times[:,0],predict_times[:,2],label="Ulepszona metoda intuicyjna")
    ax.plot(predict_times[:,0],predict_times[:,3],label="Metoda najmniejszych wariancji")
    ax.plot(predict_times[:,0],predict_times[:,4],label="Metoda Karnin'a")
    plt.title("Czasy trwania klasyfikacji dla zbioru " + name)
    plt.xlabel("Procent usuniętych połączeń")
    plt.ylabel("Czas klasyfikacji [s]")
    plt.xlim(0, 100)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + pos.height * 0.25, pos.width, pos.height * 0.75])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
    plt.savefig("wyniki/"+name+"_czasy_klasyfikacji.png")

    #generowanie kodów LaTeX dla tabel
    accuracies_data.to_latex("wyniki/"+name+"_dokładności_latex.txt", index=False, bold_rows=True, column_format="|c|c|c|c|c|c|")
    times_data.to_latex("wyniki/"+name+"_czasy_latex.txt", index=False, bold_rows=True, column_format="|c|c|c|c|c|")
    predict_times_data.to_latex("wyniki/"+name+"_czasy_klasyfikacji_latex.txt", index=False, bold_rows=True, column_format="|c|c|c|c|c|")