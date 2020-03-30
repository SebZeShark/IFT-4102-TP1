import math
class DistributionNormal:
    def __init__(self):
        self.data = []
        self.moyenne = 0
        self.deviation = 0

    def addData(self, cell):
        self.data.append(cell)

    def initialize(self):
        m = sum(self.data) / len(self.data)
        d = math.sqrt(sum([((x - m)**2) for x in self.data]) / (len(self.data) - 1))
        self.moyenne = m
        self.deviation = d

    def solve(self, x):
        var = float(self.deviation) ** 2
        denom = (2 * math.pi * var) ** .5
        num = math.exp(-(float(x) - float(self.moyenne)) ** 2 / (2 * var))
        return num / denom

    def __str__(self):
        return "Means: {}, Std Deviation {}".format(self.moyenne, self.deviation)

if __name__ == "__main__":
    test = DistributionNormal()
    test.moyenne = 5.023333333333334
    test.deviation = 0.3883504756711687
    print(test)
    var = test.solve(4.0)
    print(var)