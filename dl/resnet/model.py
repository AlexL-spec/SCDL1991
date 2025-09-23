from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_resnet50_model(num_classes:int,
                         input_shape:tuple[int,int,int]=(224,224,3),
                         base_trainable:bool|int=False,
                         dropout:float=0.3) -> keras.Model:
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    if isinstance(base_trainable, bool):
        base.trainable = base_trainable
    else:
        for layer in base.layers[:-base_trainable]:
            layer.trainable = False
        for layer in base.layers[-base_trainable:]:
            layer.trainable = True

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inputs, outputs, name="CellTypeResNet50")
    return model


__all__ = ["build_resnet50_model"]
