import numpy as np
import tensorflow as tf
from PIL import Image
#import matplotlib.pyplot as plt
import os,argparse
import collections,sys,csv,shutil,random
import cv2


self.camFolder  = os.path.join(saveFolder,"camImage")
    if not os.path.exists(self.camFolder):
        os.makedirs(self.camFolder)


def gradCam(self,conv_x,fc_x,fc_target_idx,iBatch=0):
    with tf.variable_scope("gradCam"):
        nTotCls = [int(x) for x in fc_x[iBatch].get_shape()][0]
        onehot = tf.sparse_to_dense([fc_target_idx],[nTotCls],1.0)
        loss   = tf.reduce_mean(fc_x * onehot)
        #loss   = tf.reduce_mean((fc_x - onehot)**2) # almost the same results... But less controllable as it eliminates +/- information
        grads = tf.gradients(loss,[conv_x])[0]
        grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

        outputs = conv_x[iBatch]
        gradval = grads[iBatch]

        weights = tf.reduce_mean(gradval, axis=(0,1)) # weights is calculated by each feature layer
        s1,s2,s3 = [int(x) for x in outputs.get_shape()]
        cam     = tf.zeros((s1,s2),dtype=tf.float32)

        for i in range(s3):
            cam     += weights[i] * outputs[:,:,i]

        cam = tf.nn.relu(cam)

        return cam

def gradCamGenImage(self,img,cam):
    cam  = np.expand_dims(cam,axis=2)
    cam  = cv2.resize(cam,(self.inputSize[0],self.inputSize[1]))
    cam /= np.max(cam)
    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

    img  = np.expand_dims(img,axis=2)
    img  = np.tile(img,[1,1,3])
    img /= np.max(img)
    img *= 255.

    ipzImg = img + cam * 0.5

    return ipzImg
