import tensorflow as tf
import datasets
import numpy as np

def load_and_predict(path):
    nero_net = tf.keras.models.load_model('saved_nero_net')
    # ds, ds_val = datasets.get_parameter()
    # test = next(iter(ds_val))
    # print(type(test))
    # out = nero_net.predict(test)
    # print(out)
    image = tf.io.read_file('dataset/test/bridge/File1.jpg')
    # image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image /= 255
    x = np.array([image])
    out = nero_net.predict(x)
    print(out)


if __name__ == '__main__':
    load_and_predict('')