# src/dl/efficientnetb0/model.py
"""Backbone & head builder for the EfficientNetB0-based classifier."""
from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


def build_efficientnetb0_model(num_classes: int,
                               input_shape: tuple[int, int, int] = (224, 224, 3),
                               base_trainable: bool | int = False,
                               dropout: float = 0.3) -> keras.Model:
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    if isinstance(base_trainable, bool):
        base.trainable = base_trainable
        print(f"  EfficientNetB0 base.trainable set to: {base_trainable}")
    elif isinstance(base_trainable, int) and base_trainable >= 0:
        if base_trainable == 0:
            print(f"  Freezing all layers of EfficientNetB0 base.")
            base.trainable = False
        elif len(base.layers) >= base_trainable:
            print(f"  Unfreezing last {base_trainable} layers of EfficientNetB0 base.")
            for layer in base.layers:
                layer.trainable = False
            for layer in base.layers[-base_trainable:]:
                layer.trainable = True
        else:
            print(
                f"  Warning: base_trainable_setting ({base_trainable}) >= num layers in EfficientNetB0 base ({len(base.layers)}). Unfreezing all base layers.")
            base.trainable = True
    else:
        print(f"  Invalid base_trainable_setting: {base_trainable}. Freezing all layers of EfficientNetB0 base.")
        base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    is_any_layer_in_base_trainable = any(layer.trainable for layer in base.layers)
    x_base = base(x, training=is_any_layer_in_base_trainable)

    if dropout > 0 and dropout < 1:
        x_final = layers.Dropout(dropout)(x_base)
    else:
        x_final = x_base

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x_final)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CellTypeEfficientNetB0")
    return model


__all__ = ["build_efficientnetb0_model"]