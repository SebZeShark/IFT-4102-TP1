import load_datasets as ld
from Knn import Knn

def main():
    a = ld.load_iris_dataset(0.7)
    cl = Knn(train=a[0], train_labels=a[1])
    cl.train_test(a[2], a[3])

if __name__ == "__main__":
    main()