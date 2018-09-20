#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:41:24 2017

@author: dmorton
"""

import os
import random

train_cats_dir = './catdog/train/cats/'
train_dogs_dir = './catdog//train/dogs/'

valid_cats_dir = './catdog//valid/cats/'
valid_dogs_dir = './catdog//valid/dogs/'

#%%
train_cats = os.listdir(train_cats_dir)
train_dogs = os.listdir(train_dogs_dir)
#%%

valid_cats = [x for x in sorted(random.sample(train_cats, 3125))]
valid_dogs = [x for x in sorted(random.sample(train_dogs, 3125))]

#%%
for f in valid_cats:
    os.rename(train_cats_dir + f, valid_cats_dir + f)
    
for f in valid_dogs:
    os.rename(train_dogs_dir + f, valid_dogs_dir + f)