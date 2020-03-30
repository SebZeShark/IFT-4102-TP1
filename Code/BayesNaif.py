import numpy
from MathHelper import DistributionNormal

class BayesNaifClassifier:

    def __init__(self, kwargs):
        self.case = kwargs[0]

    def train(self, train, train_labels):
        if(self.case == 0):
            self.classificateur = HouseVotesCase(train, train_labels)
        elif(self.case == 1)   :
            self.classificateur = BezdekIrisCase(train, train_labels)
        self.test(train, train_labels)




    def predict(self, exemple, label):
        return self.classificateur.predict(exemple, label)

    def test(self, test, test_labels):
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
            print("\nClasse: {}\n".format(label))
            print("Matrice de confusion: " + str(m))
            print("Accuracy: " + str((m['TP'] + m['TN']) / (m['TP'] + m['TN'] + m['FP'] + m['FN'])))
            print("Precision: " + str(m['TP'] / (m['TP'] + m['FP'])))
            print("Recall: " + str(m['TP'] / (m['TP'] + m['FN'])))
            print("\n -----------------------------------")



class Strategie:
    def predict(self, exemple, label):
        pass

    def test(self, test, test_labels):
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
        pRepublican = 1
        pDemocrat = 1
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
        self.table = [[DistributionNormal() for x in range(3)] for y in range(4)]
        for x in range(len(train)):
            case = train[x]
            index = 0 if train_labels[x] == "Iris-setosa" else 1 if train_labels[x] == "Iris-versicolor" else 2
            for y in range(len(case)):
                value = case[y]
                self.table[y][index].addData(value)
        for x in self.table:
            print()
            for y in x:
                y.initialize()


    def predict(self, exemple, label):
        pSetosa = 1
        pVersicolor = 1
        pVirginica = 1
        for i in range(len(exemple)):
            value = exemple[i]
            pS = self.table[i][0].solve(value)
            pVe = self.table[i][1].solve(value)
            pVi = self.table[i][0].solve(value)
            pSetosa *= pS
            pVersicolor *= pVe
            pVirginica *= pVi
        isSetosa = pSetosa / (pSetosa + pVersicolor + pVirginica)
        isVersicolor = pVersicolor / (pSetosa + pVersicolor + pVirginica)
        isViginica = pVirginica / (pSetosa + pVersicolor + pVirginica)
        t = [isSetosa, isVersicolor, isViginica]
        result = t.index(max(t))
        answer = "Iris-setosa" if result == 0 else "Iris-versicolor" if result == 1 else "Iris-virginica"
        warning = "========== {}, {}, {}".format(isSetosa, isVersicolor, isViginica) if answer != label else ""
        print(answer, label, warning)
        return answer == label, answer