# coding:utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import tensorflow as tf
import itchat
from itchat.content import *
import random
import logging
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as mping
from vgg19 import vgg19
import time

# loading cofigurations
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


## -------------- chotbot picture ---------------#
logging.info('loading feature mat and kdt model for chatbot-picture')
# shape(feature_mat) = (8084, 25088), <class 'numpy.ndarray'>
feature_mat = np.load('feature_mat.npy')
kdt = KDTree(feature_mat, leaf_size=40, metric='euclidean')
logging.info('loading images')
path = 'results'
total_files = []
num_results = 32
for i in range(1,num_results+1):
    dir = os.path.join(path, 'result'+str(i))
    files = os.listdir(dir)
    for file in files:
        file = os.path.join(dir,file)
        total_files.append(file)


# reply to friends, group or public account
@itchat.msg_register([TEXT,PICTURE,VIDEO], isFriendChat=True, isGroupChat=False, isMpChat=True)
def wechat(msg):
    '''
    reply according to different types of messages
    :param msg: msg can be text and picture, for other types of message, reply '爸爸！'
    :return: response to wechat
    '''
    if msg['Type'] == 'Text':
        return '嘻嘻嘻'


    elif msg['Type'] == 'Picture':
        wechat_pic(msg)

    else:
        return '傻儿子'


def wechat_pic(msg):
    '''
    get the response to image according to feature mat, save the image, and send a similar
    expression pack
    :param msg: the type of the msg is 'Picture'
    :return: response to picture
    '''
    time1 = time.time()
    sess = tf.InteractiveSession()
    mean_pixel = [123.68, 116.779, 103.939]
    # save the image
    msg['Text']('conversion_results/weichat_' + msg['FileName'])
    imag = mping.imread('conversion_results/weichat_' + msg['FileName'])
    if len(imag.shape) == 3:
        # for gif image, the third dimension is 4 and for static image, is 3
        imag = imag[:,:,:3]
        imag = imag - mean_pixel
        # cast into type of tensor, and extract features with trained vgg19
        imag = tf.cast(imag, tf.float32)
        imag = tf.image.resize_image_with_crop_or_pad(imag, 200, 200)
        imag = tf.reshape(imag, [1,200,200,3])
        time2 =time.time()
        features = vgg19(imag)
        time3 = time.time()
        features = tf.reshape(features,(1,-1))
        features = features.eval()
        # The most similar 20 expressions package index
        time4 = time.time()
        index = kdt.query(features, k=3, return_distance=False)
        time5 = time.time()
        # choose randomly except for the most similar expression
        expression_index = random.choice(index[0][1:])
        expression_file = total_files[expression_index]
        time6 = time.time()
        itchat.send('@img@%s' % expression_file, toUserName=msg['FromUserName'])
        time7 = time.time()
        print('initialize time:', time2 - time1)
        print('vggnet time:', time3 - time2)
        print('kdtree time:', time5 - time4)
        print('index time:', time6 - time5)
        print('send time:', time7 - time6)

itchat.auto_login(hotReload=True)
itchat.run()