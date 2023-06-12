# Meta-CoSTNet
Meta-learning Multiple Transportation Pattern for Traffic Demand Prediction

## Getting Started
### Data
Please refer to the files under preprocess/.

### Environment
``` 
conda create -n $ENV_NAME$ python=3.7
conda activate $ENV_NAME$

# Install compatible PyTorch 1.11.0, for example, for CUDA 11.3
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
# Or, CUDA 10.2 
pip install torch==1.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102 

pip install -r requirements.txt
```
### Saved models
You can download our trained [meta auto-encoder models](https://drive.google.com/file/d/1yljGnvdTIKIlROFCvcmT5IlnHgErOkRv/view?usp=drive_link) and [demand prediction model](https://drive.google.com/file/d/1xoh-ZCbyMW8fwOysyYuMq3930dTXCEGb/view?usp=drive_link).
Place them under the directory results/saved_models/
```
costnet/ 
    results/saved_models/
        MLAEN_1/
            bike_drop.pth
            bike_pickup.pth
            taxi_drop.pth
            taxi_pickup.pth
        Ours_best/
            best.pth
    preprocess/
    costnet/
        ...
```

## Run models
### Train our model
```
python train.py --model AutoEncoder --trainer MLAE --ddir ../datasets/ --dname multi-nyc-ae --config MLAE_config
python train.py --model Ours --trainer CoSTNet --ddir ../datasets/ --dname multi-nyc-lstm --config Ours_config
```
You can also train the baseline models, by just changing the arguments `model`. Delte the argument `trainer`. Similarly, the code loads `$MODEL_NAME$_config.py` by default. Below lists the available baseline models.

- MLP
- ConvLSTM
- STResNet

### Test our model
```
python test.py --model Ours --trainer CoSTNet --ddir ../datasets/ --dname multi-nyc-lstm --config Ours_config --ckpt ../results/saved_models/Ours_best/best.pth
```
Similarly, you can also test the baseline models, by just changing the arguments.


### Train and test CoSTNet 
```
python train.py --model AutoEncoder --ddir ../datasets/ --dname multi-nyc-ae 
python train.py --model CoSTNet --ddir ../datasets/ --dname multi-nyc-lstm 
python test.py --model CoSTNet --ddir ../datasets/ --dname multi-nyc-lstm --ckpt $CKPT_PATH$
```
