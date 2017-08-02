import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as mping
import os


def vgg19(input_image):
    '''
    extract feature from input image with trained vggnet
    :param input_image: A 4-D tensor of shape [batch, in_height, in_width, in_channels],
           the in_channels must be 3
    :return: A 4-D Tensor, extracted fearures
    '''
    VGG19_LAYERS = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'nv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    params = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    weights = params['layers'][0]
    mean = params['normalization'][0][0][0]

    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = tf.nn.conv2d(current, tf.constant(kernels), strides=(1, 1, 1, 1), padding='SAME')
            current = tf.nn.bias_add(current, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = tf.nn.max_pool(current, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    return current


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    imags = []
    features = []
    # the trained vgg19 parameters
    data_path = 'imagenet-vgg-verydeep-19.mat'
    mean_pixel = [123.68, 116.779, 103.939]
    num_results = 32
    count = 1
    for i in range(1,num_results+1):
        dir = 'results/result'+str(i)
        files = os.listdir(dir)
        batch_size = 200
        for file in files:
            file = os.path.join(dir,file)
            # type = np.array
            imag = mping.imread(file)
            if len(imag.shape) == 3:
                imag = imag[:,:,:3]
                imag = imag - mean_pixel
                # cast into type of tensor
                imag = tf.cast(imag, tf.float32)
                imag = tf.image.resize_image_with_crop_or_pad(imag, 200, 200)
                imags.append(imag)
                if count % batch_size == 0:
                    batch_images = tf.stack(imags, axis=0)
                    feature = tf.reshape(vgg19(batch_images),[batch_size,-1])
                    # convert type 'tensor' to np.array
                    features.append(feature.eval())
                    imags = []
                print('{} finished'.format(count))
                count += 1

    # The remaining pictures
    if len(imags) != 0:
        feature = tf.reshape(vgg19(imags),[len(imags),-1])
        features.append(feature.eval())

    feature_mat = np.concatenate(features)
    np.save('feature_mat_new.npy', feature_mat)

