from arg_parser import get_arguments
from data import IAMDatasetGenerator, get_dataset_parameters
from model import build_model
import keras
import tensorflow as tf

args = get_arguments()

img_size = (args.width, args.height)

train_generator = IAMDatasetGenerator('train', args.batch_size, img_size, args.debug)
val_generator = IAMDatasetGenerator('val', args.batch_size, img_size, args.debug)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices((train_generator.paths, train_generator.labels)).map(
        train_generator.process_images_labels, num_parallel_calls=AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_generator.paths, val_generator.labels)).map(
        val_generator.process_images_labels, num_parallel_calls=AUTOTUNE)

train = train_dataset.batch(args.batch_size).cache().prefetch(AUTOTUNE)
val = val_dataset.batch(args.batch_size).cache().prefetch(AUTOTUNE)

model = build_model(train_generator.chars, img_size)
model.summary()

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log_dir', histogram_freq=1)
checkpoint_filepath = 'models_dir/{epoch}'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filepath=checkpoint_filepath,
    save_freq="epoch",
    mode='max')
    
history = model.fit(
    train,
    validation_data=val,
    epochs=args.epochs,
    callbacks=[tensorboard_callback, model_checkpoint_callback],
)