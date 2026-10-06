"""
Stage 1: CNN baseline — frame-level real/fake classifier using transfer
learning on EfficientNetB0.

Why EfficientNetB0 specifically: it's a strong accuracy-per-compute
tradeoff, which matters for a solo project without heavy GPU access.
ResNet50 or MobileNetV2 are equally defensible choices — the important
thing in an interview is explaining WHY you picked one, not that you
picked the "best" one.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = (224, 224)


def build_cnn_baseline(fine_tune_layers: int = 0) -> tf.keras.Model:
    """
    fine_tune_layers: number of layers at the top of the base model to
    unfreeze for fine-tuning. 0 means the pretrained base is fully frozen
    and only the new classification head is trained — the safest starting
    point with limited data, since it can't overfit the pretrained weights.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False

    if fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # 1 = fake, 0 = real

    model = models.Model(inputs, outputs, name="deepfake_cnn_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_feature_extractor(trained_cnn: tf.keras.Model) -> tf.keras.Model:
    """
    Used in Stage 2: strips the final classification layers off a trained
    CNN so we can pull out the intermediate feature vector for each frame,
    which is what actually feeds into the LSTM (not the CNN's final
    real/fake prediction — the richer feature representation underneath it).
    """
    feature_layer = trained_cnn.layers[-3].output  # output of the Dense(128) layer
    return models.Model(inputs=trained_cnn.input, outputs=feature_layer)


if __name__ == "__main__":
    model = build_cnn_baseline()
    model.summary()
