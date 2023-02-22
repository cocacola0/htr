import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow import keras
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from data import IAMDatasetGenerator
from keras.models import load_model

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', default = 'models_dir/52/')
parser.add_argument('--out_dir', default = 'examples/')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--width', default = 128)
parser.add_argument('--height', default = 32)

args = parser.parse_args()

AUTOTUNE = tf.data.AUTOTUNE

test_generator = IAMDatasetGenerator('test', 1, (args.width, args.height), args.debug)


test_dataset = tf.data.Dataset.from_tensor_slices((test_generator.paths, test_generator.labels)).map(
        test_generator.process_images_labels, num_parallel_calls=AUTOTUNE)


test_dataset = test_dataset.batch(2).cache().prefetch(AUTOTUNE)

model = load_model(args.model_path, compile=False)

opt = keras.optimizers.Adam()

model.compile(optimizer=opt)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :test_generator.max_length
    ]

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(test_generator.int_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

index = 0

for sample in test_dataset:
    image = sample['image']
    label = sample['label']
    
    #print(tf.reduce_sum(image).numpy())
    preds = prediction_model.predict(image)
    pred_texts = decode_batch_predictions(preds)
    
    out_img = image[0]
    out_text = pred_texts[0]

    print('out_text', out_text)
    
    plt.title(out_text)
    plt.imshow(out_img, cmap='gray')
    plt.savefig(args.out_dir + str(index) + '.png')

    index += 1