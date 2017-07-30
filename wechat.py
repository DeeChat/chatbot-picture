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

# loading cofigurations
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


## -------------- chotbot picture ---------------#
logging.info('loading feature mat and kdt model for chatbot-picture')
feature_mat = np.load('feature_mat.npy')
kdt = KDTree(feature_mat, leaf_size=40, metric='euclidean')
logging.info('loading images')
total_files = []
num_results = 32
for i in range(1,num_results+1):
    dir = '/home/johnson/PycharmProjects/chatbot-picture/results/result'+str(i)
    files = os.listdir(dir)
    for file in files:
        file = dir +'/'+file
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
    # save the image
    msg['Text'](msg['FileName'])
    imag = mping.imread(msg['FileName'])
    if len(imag.shape) == 3:
        imag = imag[:,:,:3]
        # cast into type of tensor, and extract features with trained vgg19
        imag = tf.cast(imag, tf.float32)
        imag = tf.image.resize_image_with_crop_or_pad(imag, 200, 200)
        imag = tf.reshape(imag, [1,200,200,3])
        features = vgg19(imag)
        # The most similar 20 expressions package index
        index = kdt.query(features, k=20, return_distance=False)
        # choose randomly except for the most similar expression
        expression_index = random.choice(index[1:])
        expression_file = total_files[expression_index]
        itchat.send('@img@%s' % expression_file, toUserName=msg['FromUserName'])


itchat.auto_login(hotReload=True)
itchat.run()