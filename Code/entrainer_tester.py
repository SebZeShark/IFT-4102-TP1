import numpy as np
import sys
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
#import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initializer vos paramètres

param = ["bezdelIris"]



# Initializer/instanciez vos classifieurs avec leurs paramètres

classificateur = BayesNaif.BayesNaifClassifier(param)



# Charger/lire les datasets

train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.6)




# Entrainez votre classifieur
classificateur.train(train, train_labels)



# Tester votre classifieur
classificateur.test(test, test_labels)





