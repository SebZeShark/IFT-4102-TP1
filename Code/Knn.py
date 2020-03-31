"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas, 
    * train 	: pour entrainer le modèle sur l'ensemble d'entrainement
    * predict 	: pour prédire la classe d'un exemple donné
    * test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes test, predict et test de votre code.
"""

from collections import Counter


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Knn:

    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.train_list = kwargs['train']
        self.train_labels = kwargs['train_labels']
        self.dist_equation = kwargs['dist_equation']

    def dist(self, e1, e2):
        return sum(self.dist_equation(x, y) for x, y in zip(e1, e2))

    def predict(self, exemple, label, k=5):
        """
        Prédire la classe d'un exemple donné en entrée
        exemple est de taille 1xm

        si la valeur retournée est la meme que la veleur dans label
        alors l'exemple est bien classifié, si non c'est une missclassification

        """
        distances = []
        for i in range(self.train_labels.size):
            distances.append((self.dist(self.train_list[i], exemple), self.train_labels[i]))

        distances.sort(key=lambda x: x[0])  # on classe les items d'entrainement selon leur distance avec l'exemple
        classes = map(lambda y: y[1], distances[:k])
        class_found = Counter(classes).most_common(1)[0][0]
        # puis on choisit la classe de la pluralité des k plus petites distances
        return class_found == label, class_found

    def train_test(self, test, test_labels, header=""):
        """
        c'est la méthode qui va tester votre modèle sur les données de test
        l'argument test est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        test_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        Faites le test sur les données de test, et afficher :
        - la matrice de confision (confusion matrix)
        - l'accuracy
        - la précision (precision)
        - le rappel (recall)

        Bien entendu ces tests doivent etre faits sur les données de test seulement

        """
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
            print("\n" + header)
            print("Classe: {}\n".format(label))
            print("Matrice de confusion: " + str(m))
            print("Accuracy: " + str(((m['TP'] + m['TN'])/(m['TP'] + m['TN'] + m['FP'] + m['FN']) )
                                     if (m['TP'] + m['TN'] + m['FP'] + m['FN']) else 0))
            print("Precision: " + str((m['TP'] / (m['TP'] + m['FP'])) if (m['TP'] + m['FP']) else 0 ))
            print("Recall: " + str((m['TP'] / (m['TP'] + m['FN'])) if (m['TP'] + m['FN']) else 0))
            print("\n -----------------------------------")



    # Vous pouvez rajouter d'autres méthodes et fonctions,
    # il suffit juste de les commenter.