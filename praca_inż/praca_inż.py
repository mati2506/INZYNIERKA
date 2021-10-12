import numpy as np
import math as mt
import matplotlib.pyplot as plt
#from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class my_MLP(object):
    def __init__(self, hidden=(10, 10, 10), epochs=100, eta=0.1, shuffle=True):
        self.hidden = hidden    #Liczba neuronów na kolejnych warstwach ukrytych
        self.hidden_count = len(hidden) #Liczba powłok ukrytych
        self.epochs = epochs    #Liczba epok
        self.eta = eta          #Współczynnik uczenia
        self.shuffle = shuffle  #Czy mieszać próbki w epokach

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
        for i in range(self.hidden_count):
            self.weight_hidden.append(np.random.normal(0,0.1,size=(self.feature_count, self.hidden)))
            self.bias_hidden.append(np.zeros(self.hidden))
        self.weight_out = np.random.normal(0,0.1,size=(self.hidden, self.class_count))
        self.bias_out = np.zeros(self.class_count)

        for i in range(self.epochs):
            indexes = np.array(range(self.samples_count))

            if self.shuffle == True:
                indexes = shuffle(indexes)

            for ind in indexes:
                activation_hidden, activation_out = self._forward(X[j])

        #...

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





#    def _forward(self, X):
#        out_u = np.dot(X, self.w_h) + self.b_h
#        aktywacja_ukryta = self._sigmoid(out_u)
#        out_f = np.dot(aktywacja_ukryta, self.w_out) + self.b_out
#        aktywacja_koncowa = self._sigmoid(out_f)
#        return aktywacja_ukryta, aktywacja_koncowa

#    def _compute_cost(self, y, output):
#        J = 0
#        for i in range(self.liczba_probek):
#            for k in range(self.liczba_klas):
#                J = J - (y[i][k]*mt.log(output[i][k])+(1-y[i][k])*mt.log(1-output[i][k]))

#        return J

#    def dokladnosc(self, y_r, y_w):
#        liczba = y_r.shape[0]
#        licznik = 0
#        for i in range(liczba):
#            czy_rozne = 0
#            for j in range(y_r.shape[1]):
#                if y_r[i][j] != y_w[i][j]:
#                    czy_rozne = 1
#                    break
#            if czy_rozne == 0:
#                licznik = licznik+1
#        wynik = (licznik*100)/liczba
#        return wynik

#    def fit(self, X, y):
#        self.liczba_probek = X.shape[0]
#        self.liczba_cech = X.shape[1]
#        self.liczba_klas = y.shape[1]
#        self.w_h = np.random.normal(0,0.1,size=(self.liczba_cech, self.hidden))
#        self.b_h = np.zeros(self.hidden)
#        self.w_out = np.random.normal(0,0.1,size=(self.hidden, self.liczba_klas))
#        self.b_out = np.zeros(self.liczba_klas)

#        koszt_epoki = []
#        dokladnosc_epoki = []

#        for i in range(self.epochs):
#            ind = np.array(range(self.liczba_probek))

#            if self.shuffle == True:
#                ind = shuffle(ind)
            
#            for j in ind:
#                out_u, out_f = self._forward(X[j])
#                pochodna_out_f = out_f*(1-out_f)
#                delta_out = (out_f - y[j])*pochodna_out_f
#                pochodna_out_u = out_u*(1-out_u)
#                delta_h = np.dot(delta_out, np.transpose(self.w_out))*pochodna_out_u
#                gradient_wag_h = np.outer(X[j], delta_h)
#                gradient_biasow_h = delta_h
#                gradient_wag_out = np.outer(out_u, delta_out)
#                gradient_biasow_out = delta_out

#                self.w_h = self.w_h - self.eta*gradient_wag_h
#                self.b_h = self.b_h - self.eta*gradient_biasow_h
#                self.w_out = self.w_out - self.eta*gradient_wag_out
#                self.b_out = self.b_out - self.eta*gradient_biasow_out
            
#            wyj_p, wyj_k = self.predict(X)
#            koszt_epoki.append(self._compute_cost(y,wyj_p))
#            dokladnosc_epoki.append(self.dokladnosc(y,wyj_k))

#        return np.array(koszt_epoki), np.array(dokladnosc_epoki)

#    def predict(self, X): #zwraca: [0] - prediction; [1] - klasy
#        licz_probek = X.shape[0]

#        pred = []
#        klasy = []

#        for i in range(licz_probek):
#            _, praw = self._forward(X[i])
#            pred.append(praw)
            
#            index = np.argmax(praw)
#            wektor = np.zeros(self.liczba_klas)
#            wektor[index] = 1
#            klasy.append(wektor)
        
#        return np.array(pred), np.array(klasy)

#if __name__ == '__main__':
#    X_iris, y_iris = fetch_openml(name="iris", version=1, return_X_y=True)

#    y_iris_coded=[]
#    for i in range(len(y_iris)):
#        if y_iris[i] == 'Iris-setosa':
#            y_iris_coded.append([1.,0.,0.])
#        elif y_iris[i] == 'Iris-versicolor':
#            y_iris_coded.append([0.,1.,0.])
#        else:
#            y_iris_coded.append([0.,0.,1.])

#    y_iris_coded = np.array(y_iris_coded)

#    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris_coded, random_state=13)
    
#    mlp1 = MLP()
#    koszt, dokladnosc = mlp1.fit(X_train, y_train)

#    print("Zbiór iris:")
#    print("Koszty w kolejnych epokach:")
#    print(koszt)
#    print()
#    print("Dokładności w kolejnych epokach:")
#    print(dokladnosc)
#    print()
#    plt.plot(koszt, '*-')
#    plt.title("Koszty w kolejnych epokach zbioru Iris")
#    plt.figure()
#    plt.plot(dokladnosc, '*-')
#    plt.title("Dokładności w kolejnych epokach zbioru Iris")

#    _, y_pred = mlp1.predict(X_test)

#    dokladnosc_test = mlp1.dokladnosc(y_test, y_pred)
#    print("Dokładność klasyfikacji zbioru testowego:")
#    print(dokladnosc_test)
#    print()
#    #plt.show()

#    mlp1skal=MLP()
#    scaler1 = MinMaxScaler(feature_range=(0, 1))
#    scaled1=scaler1.fit_transform(X_train)
#    kosztskal, dokladnoscskal=mlp1skal.fit(scaled1,y_train)
    
#    print("Koszty w kolejnych epokach przeskalowane:")
#    print(kosztskal)
#    print()
#    print("Dokładności w kolejnych epokach przeskalowane:")
#    print(dokladnoscskal)
#    print()
#    plt.figure()
#    plt.plot(kosztskal, '*-')
#    plt.title("Koszty w kolejnych epokach zbioru Iris przeskalowanego")
#    plt.figure()
#    plt.plot(dokladnoscskal, '*-')
#    plt.title("Dokładności w kolejnych epokach zbioru Iris przeskalowanego")
#    plt.show()

#    print()


#    X, y = make_classification(n_samples=250, random_state=13)

#    Y = []
#    for i in range(y.shape[0]):
#        if y[i] == 0:
#            Y.append(np.array([1,0]))
#        else:
#            Y.append(np.array([0,1]))
#    Y = np.array(Y)

#    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y, random_state=13)
    
#    mlp2 = MLP()
#    koszt2, dokladnosc2 = mlp2.fit(X_train2, y_train2)

#    print("Zbiór binarny:")
#    print("Koszty w kolejnych epokach:")
#    print(koszt2)
#    print()
#    print("Dokładności w kolejnych epokach:")
#    print(dokladnosc2)
#    print()
#    plt.figure()
#    plt.plot(koszt2, '*-')
#    plt.title("Koszty w kolejnych epokach zbioru binarnego")
#    plt.figure()
#    plt.plot(dokladnosc2, '*-')
#    plt.title("Dokładności w kolejnych epokach zbioru binarnego")

#    _, y_pred2 = mlp2.predict(X_test2)

#    dokladnosc_test2 = mlp1.dokladnosc(y_test2, y_pred2)
#    print("Dokładność klasyfikacji zbioru testowego:")
#    print(dokladnosc_test2)
#    print()
#    #plt.show()

#    mlp2skal=MLP()
#    scaler2 = MinMaxScaler(feature_range=(0, 1))
#    scaled2=scaler2.fit_transform(X_train2)
#    koszt2skal, dokladnosc2skal=mlp2skal.fit(scaled2,y_train2)
    
#    print("Koszty w kolejnych epokach przeskalowane:")
#    print(koszt2skal)
#    print()
#    print("Dokładności w kolejnych epokach przeskalowane:")
#    print(dokladnosc2skal)
#    plt.figure()
#    plt.plot(koszt2skal, '*-')
#    plt.title("Koszty w kolejnych epokach zbioru binarnego przeskalowanego")
#    plt.figure()
#    plt.plot(dokladnosc2skal, '*-')
#    plt.title("Dokładności w kolejnych epokach zbioru binarnego przeskalowanego")
#    plt.show()