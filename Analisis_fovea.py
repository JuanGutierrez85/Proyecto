import numpy as np
import pandas
import matplotlib.pyplot as plt
from statistics import mean, median, stdev

df= pandas.read_csv("props.csv",sep=',', names=["p", "v"])


prof = df[['p']]
prof = prof.to_numpy()
pro = np.array([])
for i in range(len(prof)):
    pro=np.append(pro,prof[i])
vol = df[['v']]
vol = vol.to_numpy()
vo = np.array([])
for i in range(len(vol)):
    vo=np.append(vo,vol[i])
print(pro)
print(vo)

pro = np.sort(pro)
print(pro)
vo = np.sort(vo)
print(vo)
n=np.array([])
for i in range(len(prof)):
    n=np.append(n,i+1)

plt.scatter(n,pro)
plt.show()
plt.hist(pro,15)
plt.show()
plt.boxplot(pro, meanline=True, showmeans=True,vert=False)
plt.show()
statspro = mean(pro), median(pro), stdev(pro)
print(statspro)

plt.scatter(n,vo)
plt.show()
plt.hist(vo,15)
plt.show()
plt.boxplot(vo, meanline=True, showmeans=True, vert=False)
plt.show()
statsvo = mean(vo), median(vo), stdev(vo)
print(statsvo)

