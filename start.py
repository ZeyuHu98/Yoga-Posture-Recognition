import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, datasets, losses, layers, Sequential
import numpy as np
import datasets


def run():
    nero_net = keras.applications.ResNet50(include_top=True, weights=None,
                                           pooling='Max', classes=4)
    # 输入格式
    nero_net.build(input_shape=[None, 224, 224, 3])
    nero_net.summary()
    # 装配
    nero_net.compile(optimizer=optimizers.Adam(lr=0.001),
                    loss=losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                    )
    # 数据预处理 获取数据集
    ds, ds_val = datasets.get_parameter()
    # 输入数据集 标签集 训练次数 验证频率 进行训练
    nero_net.fit(ds, epochs=5, validation_data=ds_val, validation_freq=2)

    # 评估训练结果
    nero_net.evaluate(ds_val)
    # 保存模型
    tf.saved_model.save(nero_net, 'saved_nero_net')
    # 预测
    test = next(iter(ds_val))
    out = nero_net.predict(test)
    print(out)

if __name__ == '__main__':
    run()



