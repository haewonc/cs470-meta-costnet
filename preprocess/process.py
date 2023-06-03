import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm 
import argparse 
import math
from skimage import transform 

parser = argparse.ArgumentParser()
parser.add_argument('-offset', '--offset', nargs='*', default=None)
args = parser.parse_args()

bike = pd.read_csv('bike.csv')
taxi = pd.read_csv('taxi.csv')

bike = bike[['starttime','stoptime','start station latitude','start station longitude','end station latitude','end station longitude']]
taxi = taxi[['tpep_pickup_datetime','tpep_dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']]

bike.dropna(inplace=True)
taxi.dropna(inplace=True)

coords = np.array([
    [40.80, -73.934],
    [40.73, -73.982],
    [40.742, -74.008],
    [40.812, -73.96]
])
center = coords.mean(axis=0)
coords -= center
angle = np.arctan((coords[1, 1] - coords[0, 1]) / (coords[1, 0] - coords[0, 0]))
rotation_matrix = transform.EuclideanTransform(rotation=-angle)
rotated_coords = rotation_matrix(coords)
shear_angle = np.arctan((rotated_coords[2, 1] - rotated_coords[1, 1]) / (rotated_coords[2, 0] - rotated_coords[1, 0]))
shear_angle = np.pi/2 - shear_angle
shear_matrix = transform.AffineTransform(shear=shear_angle)
sheared_coords = shear_matrix(rotated_coords)

names = [
    {'pu': 'start station ', 'do': 'end station '},
    {'pu': 'pickup_', 'do': 'dropoff_'},
]
for i, df in enumerate([bike, taxi]):
    for a in ['pu', 'do']:
        b = names[i][a]
        data_coords = shear_matrix(rotation_matrix(np.array([df[f'{b}latitude'].to_numpy(), df[f'{b}longitude'].to_numpy()]).T-center))
        df[f'{a}_row'] = 20 * (data_coords[:,0] - coords[:,0].min()) / (coords[:,0].max() - coords[:,0].min())
        df[f'{a}_col'] = 20 * (data_coords[:,1] - coords[:,1].min()) / (coords[:,1].max() - coords[:,1].min())
        df[f'{a}_row'].apply(lambda x: int(x))
        df[f'{a}_col'].apply(lambda x: int(x))

        df.drop(df[df[f'{a}_row'] < 0].index, inplace=True)
        df.drop(df[df[f'{a}_col'] < 0].index, inplace=True)
        df.drop(df[df[f'{a}_row'] >= 20].index, inplace=True)
        df.drop(df[df[f'{a}_col'] >= 20].index, inplace=True)
    
bike['pu_time'] = bike['starttime']
bike['do_time'] = bike['stoptime']
bike.drop(['starttime','stoptime','start station latitude','start station longitude','end station latitude','end station longitude'], axis=1, inplace=True)

taxi['pu_time'] = taxi['tpep_pickup_datetime']
taxi['do_time'] = taxi['tpep_dropoff_datetime']
taxi.drop(['tpep_pickup_datetime','tpep_dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis=1, inplace=True)

print(bike.describe())
print(taxi.describe())

# Demand data 
TSTEPS = 91 * 24 * 2 # 30 mins, total 91 days
offsets = args.offset # to generate data for autoencoder
GSIZE = 20 # Grid size
SHAPE = 4 # bike pu, bike do, taxi pu, taxi do

bike.dropna(inplace=True)
taxi.dropna(inplace=True)

def create_data(offset):
    data = np.zeros(shape=(TSTEPS, GSIZE, GSIZE, SHAPE))
    for index, row in tqdm(bike.iterrows(), total=len(bike)):
        pu = (pd.to_datetime(row['pu_time']) - pd.Timestamp("2016-04-01") - pd.Timedelta(offset)) // pd.Timedelta("30m")
        do = (pd.to_datetime(row['do_time']) - pd.Timestamp("2016-04-01") - pd.Timedelta(offset)) // pd.Timedelta("30m")
        if pu < 0 or do < 0:
            continue
        if pu >= TSTEPS or do >= TSTEPS:
            continue
        data[pu, int(row['pu_row']), int(row['pu_col']), 0] += 1
        data[do, int(row['do_row']), int(row['do_col']), 1] += 1
    for index, row in tqdm(taxi.iterrows(), total=len(taxi)):
        pu = (pd.to_datetime(row['pu_time']) - pd.Timestamp("2016-04-01") - pd.Timedelta(offset)) // pd.Timedelta("30m")
        do = (pd.to_datetime(row['do_time']) - pd.Timestamp("2016-04-01") - pd.Timedelta(offset)) // pd.Timedelta("30m")
        if pu < 0 or do < 0:
            continue
        if pu >= TSTEPS or do >= TSTEPS:
            continue
        data[pu, int(row['pu_row']), int(row['pu_col']), 2] += 1
        data[do, int(row['do_row']), int(row['do_col']), 3] += 1
    
    data_min, data_max = [], []
    for i in range(4):
        data_min.append(data[:,:,:,i].min())
        data[:,:,:,i] -= data[:,:,:,i].min()
        data_max.append(data[:,:,:,i].max())
        data[:,:,:,i] /= data[:,:,:,i].max()
    
    return data, [data_min, data_max]

if offsets is None:
    data, [data_min, data_max] = create_data("0m")
    ext = np.load('ext_proc.npy')
    data_all = {
        "train": {
            "data": data[:-672, :, :, :],
            "ext": ext[:-672, :]
        },
        "test": {
            "data": data[-672:, :, :, :],
            "ext": ext[-672:, :]
        },
        "min": data_min, 
        "max": data_max
    }
    with open('multi-nyc-lstm.pkl', 'wb') as file:
        pickle.dump(data_all, file)
else: 
    train = []
    test = []
    for offset in offsets:
        print(f"Processing offset: {offset}")
        data, [data_min, data_max] = create_data(offset)
        train.append(data[:-672, :, :, :])
        test.append(data[-672:, :, :, :])
    
    data_all = {
        "train": np.concatenate(train, axis=0),
        "test": np.concatenate(test, axis=0),
        "min": data_min, 
        "max": data_max
    }
    with open('multi-nyc-ae.pkl', 'wb') as file:
        pickle.dump(data_all, file)
