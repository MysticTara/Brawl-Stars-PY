import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

statsFile = open("stats.csv", 'r')
stats = csv.reader(statsFile)

next(stats)
features = []
labels = []

test = [[1,1036,5320,720,1.25,5.5,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,]]


for stat in stats:
  for i in range(len(stat[1:-1])):
    stat[i + 1] = float(stat[i + 2])
  features.append(stat[1:-1])
  labels.append(stat[-1])

poly = PolynomialFeatures(degree=3)
features = poly.fit_transform(features)
test = poly.fit_transform(test)

model = LinearRegression()
model.fit(features, labels)

print(model.predict(test))