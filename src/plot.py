import matplotlib.pyplot as plt

X, Y = [], []
for line in open('../data/output.txt', 'r'):
  values = [float(s) for s in line.split()]
  X.append(values[0])

#print(X)
plt.plot(range(len(X)), X)
plt.show()