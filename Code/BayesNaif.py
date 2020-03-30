import numpy
from MathHelper import DistributionNormal

class BayesNaifClassifier:

    def __init__(self, kwargs):
        self.case = kwargs[0]

    def train(self, train, train_labels, test_name):
        if(self.case == 0):
            self.classificateur = HouseVotesCase(train, train_labels)
        elif(self.case == 1)   :
            self.classificateur = BezdekIrisCase(train, train_labels)
        else:
            self.classificateur = MonkCase(train, train_labels)
        self.test(train, train_labels, test_name)




    def predict(self, exemple, label):
        return self.classificateur.predict(exemple, label)

    def test(self, test, test_labels, test_name =""):
        metriques = {}
        for i in set(test_labels):
            metriques[i] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        for i in zip(test, test_labels):
            resultat = self.predict(i[0], i[1])
            if resultat[0]:
                for label, m in metriques.items():
                    if label == i[1]:
                        metriques[label]['TP'] += 1
                    else:
                        metriques[label]['TN'] += 1
            else:
                for label, m in metriques.items():
                    if label == i[1]:
                        metriques[label]['FN'] += 1
                    elif label == resultat[1]:
                        metriques[label]['FP'] += 1
                    else:
                        metriques[label]['TN'] += 1

        for label, m in metriques.items():
            print("\nClasse: {}\n".format(test_name))
            print("Matrice de confusion: " + str(m))
            print("Accuracy: " + str((m['TP'] + m['TN']) / (m['TP'] + m['TN'] + m['FP'] + m['FN'])))
            print("Precision: " + str(m['TP'] / (m['TP'] + m['FP'])))
            print("Recall: " + str(m['TP'] / (m['TP'] + m['FN'])))
            print("\n -----------------------------------")



class Strategie:
    def predict(self, train, train_labels):
        pass

class HouseVotesCase(Strategie):
    def __init__(self, train, train_labels):
        self.republicans = train_labels.tolist().count("republican")
        self.democrats = train_labels.tolist().count("democrat")
        self.total = self.republicans + self.democrats
        self.tables = [[[0 for x in range(2)] for y in range(3)] for z in range(16)]
        for x in range(len(train)):
            set = train[x]
            allignement = 0 if train_labels[x] == "republican" else 1
            for y in range(len(set)):
                value = set[y]
                self.tables[y][value][allignement] += 1

        self.likelyhood = [[[0 for x in range(2)] for y in range(3)] for z in range(16)]
        for x in range(len(self.tables)):
            table = self.tables[x]
            for i in range(3):
                self.likelyhood[x][i][0] = table[i][0]/self.republicans
            for i in range(3):
                self.likelyhood[x][i][1] = table[i][1]/self.democrats

    def predict(self, exemple, label):
        pRepublican = self.republicans / self.total
        pDemocrat = self.democrats / self.total
        for i in range(len(exemple)):
            value = exemple[i]
            propsR = self.likelyhood[i][value][0]
            propsD = self.likelyhood[i][value][1]
            pRepublican *= propsR
            pDemocrat *= propsD

        isRepublican = pRepublican / (pRepublican + pDemocrat)
        isDemocrat = pDemocrat / (pRepublican + pDemocrat)
        answer = "republican" if isRepublican > isDemocrat else "democrat"
        return answer == label, answer

class BezdekIrisCase(Strategie):
    def __init__(self,train, train_labels):
        self.setosa = train_labels.tolist().count("Iris-setosa")
        self.versicolor = train_labels.tolist().count("Iris-versicolor")
        self.virginia = train_labels.tolist().count("Iris-virginica")
        self.total = self.setosa + self.versicolor + self.virginia
        self.table = [[DistributionNormal() for x in range(3)] for y in range(4)]
        for x in range(len(train)):
            case = train[x]
            index = 0 if train_labels[x] == "Iris-setosa" else 1 if train_labels[x] == "Iris-versicolor" else 2
            for y in range(len(case)):
                value = case[y]
                self.table[y][index].addData(value)
        for x in self.table:
            for y in x:
                y.initialize()


    def predict(self, exemple, label):
        pSetosa = self.setosa / self.total
        pVersicolor = self.versicolor / self.total
        pVirginica = self.virginia / self.total
        for i in range(len(exemple)):
            value = exemple[i]
            pS = self.table[i][0].solve(value)
            pVe = self.table[i][1].solve(value)
            pVi = self.table[i][2].solve(value)
            #print(pS, pVe, pVi)
            pSetosa *= pS
            pVersicolor *= pVe
            pVirginica *= pVi
        isSetosa = pSetosa / (pSetosa + pVersicolor + pVirginica)
        isVersicolor = pVersicolor / (pSetosa + pVersicolor + pVirginica)
        isViginica = pVirginica / (pSetosa + pVersicolor + pVirginica)
        t = [isSetosa, isVersicolor, isViginica]
        result = t.index(max(t))
        answer = "Iris-setosa" if result == 0 else "Iris-versicolor" if result == 1 else "Iris-virginica"
        return answer == label, answer

class MonkCase(Strategie):
    def __init__(self, train, train_labels):
        self.zeros = train_labels.tolist().count("0")
        self.ones = train_labels.tolist().count("1")
        self.total = self.zeros + self.ones
        self.tables = [[[0 for x in range(2)] for y in range(4)] for z in range(6)]
        for x in range(len(train)):
            set = train[x]
            classe = 0 if train_labels[x] == "0" else 1
            for y in range(len(set)):
                value = set[y] -1
                self.tables[y][value][classe] += 1

        self.likelyhood = [[[0 for x in range(2)] for y in range(4)] for z in range(6)]
        for x in range(len(self.tables)):
            table = self.tables[x]
            for i in range(4):
                self.likelyhood[x][i][0] = table[i][0] / self.zeros
            for i in range(4):
                self.likelyhood[x][i][1] = table[i][1] / self.ones

    def predict(self, exemple, label):
        pZero = self.zeros / self.total
        pOnes = self.ones / self.total
        for i in range(len(exemple)):
            value = exemple[i] - 1
            propsZ = self.likelyhood[i][value][0]
            propsO = self.likelyhood[i][value][1]
            pZero *= propsZ if propsZ != 0 else 1
            pOnes *= propsO if propsO != 0 else 1

        isZero = pZero / (pZero + pOnes)
        isOne = pOnes / (pZero + pOnes)
        answer = "0" if isZero > isOne else "1"
        return answer == label, answer