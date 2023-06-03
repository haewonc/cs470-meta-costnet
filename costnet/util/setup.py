import argparse 
from util.logging import *
import importlib

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-trainer', '--trainer', type=str,
        help='Trainer name')
    parser.add_argument('-model', '--model', type=str, default=None,
        help='Model name')
    parser.add_argument('-ckpt', '--ckpt', type=str, default=None,
        help='Model ckpt')
    parser.add_argument('-config', '--config', default=None, type=str,
        help='Config name')
    parser.add_argument('-dataset_dir', '--ddir', type=str,
        help='Path to dataset')
    parser.add_argument('-dataset_name', '--dname', type=str,
        help='Name of the dataset')
    parser.add_argument('-device', '--device', default='cuda:0', type=str,
        help='GPU to enable (default: cuda:0)')
    args = parser.parse_args()
    if args.model == None:
        args.model = args.trainer
    if args.config == None:
        args.config = '{}_config'.format(args.trainer)
    print(toGreen('Arguments loaded succesfully'))
    return args

def load_trainer(args):
    try: 
        confg_class = getattr(importlib.import_module("config.{}".format(args.config)), args.config)
        config = confg_class(args.device, args.ddir, args.dname)
    except:
        print(toRed('Config undefined'))
        raise 
    print(toGreen('Config loaded succesfully'))
    try: 
        model_class = getattr(importlib.import_module("model.{}".format(args.model)), "{}Model".format(args.model))
        trainer_class = getattr(importlib.import_module("trainer.{}_trainer".format(args.trainer)), "{}Trainer".format(args.trainer))
        trainer = trainer_class(model_class, config, args)
        
    except:
        print(toRed('Model undefined'))
        raise 
    print(toGreen('Trainer loaded succesfully'))
    return trainer
