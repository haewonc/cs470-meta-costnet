import pandas as pd
import numpy as np 

ext = pd.read_csv('ext.csv')
data = []

cnt = 0
for index, row in ext.iterrows():
    hourly = [row['temp'], row['humidity'], int('Rain' in row['conditions']), int('Overcast' in row['conditions']), int('Partially cloudy' in row['conditions']), int('Clear' in row['conditions']), int(((index//24) % 7 == 1) or ((index//24) % 7 == 2) or ((index//24) == 59))]
    data.append(hourly)
    data.append(hourly)

data = np.array(data)

for i in [0, 1]:
    data[:,i] -= data[:,i].min() 
    data[:,i] /= data[:,i].max()

print(pd.DataFrame(data).describe())
np.save('ext_proc.npy', data)