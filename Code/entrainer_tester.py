import load_datasets as ld
from Knn import Knn
from BayesNaif import BayesNaifClassifier

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester
"""


def main():
    # Initializer vos paramètres
    i = ld.load_iris_dataset(0.7)
    c = ld.load_congressional_dataset(0.7)
    m1 = ld.load_monks_dataset(1)
    m2 = ld.load_monks_dataset(2)
    m3 = ld.load_monks_dataset(3)

    # Initializer/instanciez vos classifieurs avec leurs paramètres

    euclide = lambda x, y: pow((x - y),
                               2)  # Pas besoin d'extraire la racine, car cela ne changera pas l'ordre de classement
    diff_binaire = lambda x, y: 0 if x == y else 1

    knn_i = Knn(train=i[0], train_labels=i[1], dist_equation=euclide)
    knn_c = Knn(train=c[0], train_labels=c[1], dist_equation=euclide)
    knn_m1 = Knn(train=m1[0], train_labels=m1[1], dist_equation=diff_binaire)
    knn_m2 = Knn(train=m2[0], train_labels=m2[1], dist_equation=diff_binaire)
    knn_m3 = Knn(train=m3[0], train_labels=m3[1], dist_equation=diff_binaire)

    bn_i = BayesNaifClassifier([1])
    bn_c = BayesNaifClassifier([0])
    bn_m1 = BayesNaifClassifier([2])
    bn_m2 = BayesNaifClassifier([2])
    bn_m3 = BayesNaifClassifier([2])

    # Entrainez votre classifieur
    print("\n=============\nKNN train tests\n=============")
    knn_i.train_test(i[0], i[1], "Dataset: Iris, Training")
    knn_c.train_test(c[0], c[1], "Dataset: Congressional, Training")
    knn_m1.train_test(m1[0], m1[1], "Dataset: MONKS-1, Training")
    knn_m2.train_test(m2[0], m2[1], "Dataset: MONKS-2, Training")
    knn_m3.train_test(m3[0], m3[1], "Dataset: MONKS-3, Training")

    print("\n=============\nBayes Naif train tests\n=============")
    bn_i.train(i[0], i[1], "Dataset: Iris, Test")
    bn_c.train(c[0], c[1], "Dataset: Congressional, Test")
    bn_m1.train(m1[0], m1[1], "Dataset: MONKS-1, Test")
    bn_m2.train(m2[0], m2[1], "Dataset: MONKS-2, Test")
    bn_m3.train(m3[0], m3[1], "Dataset: MONKS-3, Test")

    print("\n=============\nKNN tests\n=============")
    # Tester votre classifieur
    knn_i.train_test(i[2], i[3], "Dataset: Iris, Test")
    knn_c.train_test(c[2], c[3], "Dataset: Congressional, Test")
    knn_m1.train_test(m1[2], m1[3], "Dataset: MONKS-1, Test")
    knn_m2.train_test(m2[2], m2[3], "Dataset: MONKS-2, Test")
    knn_m3.train_test(m3[2], m3[3], "Dataset: MONKS-3, Test")

    print("\n=============\nBayes Naif tests\n=============")
    bn_i.test(i[2], i[3], "Dataset: Iris, Test")
    bn_c.test(c[2], c[3], "Dataset: Congressional, Test")
    bn_m1.test(m1[2], m1[3], "Dataset: MONKS-1, Test")
    bn_m2.test(m2[2], m2[3], "Dataset: MONKS-2, Test")
    bn_m3.test(m3[2], m3[3], "Dataset: MONKS-3, Test")


if __name__ == "__main__":
    main()

