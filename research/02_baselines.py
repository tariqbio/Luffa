"""
02_baselines.py — Compare your model against standard baselines
Trains ResNet50, VGG16, EfficientNetB0 on your dataset with same settings.

Usage:
    python research/02_baselines.py
"""
import numpy as np
import tensorflow as tf

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR  = 'data/train'
VAL_DIR    = 'data/val'
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
NUM_CLASSES= 2
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(path):
    norm = tf.keras.layers.Rescaling(1./255)
    ds = tf.keras.utils.image_dataset_from_directory(
        path, image_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE, label_mode='int')
    return ds.map(lambda x,y:(norm(x),y)).prefetch(tf.data.AUTOTUNE)

def build_model(base_fn, name):
    base = base_fn(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable = False
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    m = tf.keras.Model(base.input, out, name=name)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return m

BASELINES = [
    ('ResNet50',       tf.keras.applications.ResNet50),
    ('VGG16',          tf.keras.applications.VGG16),
    ('EfficientNetB0', tf.keras.applications.EfficientNetB0),
    ('MobileNetV2',    tf.keras.applications.MobileNetV2),
    ('InceptionV3',    tf.keras.applications.InceptionV3),
]

if __name__ == '__main__':
    train_ds = make_dataset(TRAIN_DIR)
    val_ds   = make_dataset(VAL_DIR)
    results  = {}
    for name, fn in BASELINES:
        print(f'\n{"="*40}\nTraining {name}…\n{"="*40}')
        m = build_model(fn, name)
        cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                                monitor='val_accuracy', mode='max')]
        m.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb, verbose=1)
        loss, acc = m.evaluate(val_ds, verbose=0)
        results[name] = round(acc*100, 2)
        print(f'{name}: Val Accuracy = {acc*100:.2f}%')

    print('\n\n── Baseline Comparison ──')
    print(f'{"Model":<20} {"Val Accuracy":>14}')
    print('-'*35)
    for name, acc in sorted(results.items(), key=lambda x:-x[1]):
        print(f'{name:<20} {acc:>13.2f}%')
    print(f'{"TariqCViT (Yours)":<20} {"99.51":>13}%  ← your model')
