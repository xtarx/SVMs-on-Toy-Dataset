import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pylab as pl
import cv2

def read_file(file_name):
    data = pd.read_csv('/Users/xtarx/Documents/TUM/4th/MLMA/ex2-svm/03-digits-dataset/'+file_name)
    y=data[data.columns[0]]
    x=data.drop(data.columns[[0]], axis=1).as_matrix()
    return  x,y

train_x,train_y=read_file("train.csv")

# print((train_y).shape)
# print((train_x[0]))

# cv2.imshow('image',train_x[5])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# my_img = plt.imread(train_x[5])

pixels = train_x[5].reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

