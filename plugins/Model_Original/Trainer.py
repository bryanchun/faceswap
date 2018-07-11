
import time
import numpy
import os.path
from lib.training_data import TrainingDataGenerator, stack_images
import tensorflow as tf

class Trainer():
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4,
    }

    def __init__(self, model, fn_A, fn_B, batch_size, *args):
        self.batch_size = batch_size
        self.model = model

        generator = TrainingDataGenerator(self.random_transform_args, 160)
        self.images_A = generator.minibatchAB(fn_A, self.batch_size)
        self.images_B = generator.minibatchAB(fn_B, self.batch_size)  

    def train_one_step(self, iter, viewer, callback):
        epoch, warped_A, target_A = next(self.images_A)
        epoch, warped_B, target_B = next(self.images_B)
        
        loss_A = self.model.autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = self.model.autoencoder_B.train_on_batch(warped_B, target_B)
        print("[{0}] [#{1:05d}] loss_A: {2:.5f}, loss_B: {3:.5f}".format(time.strftime("%H:%M:%S"), iter, loss_A, loss_B),
            end='\r')
        self.write_log(callback, ['loss_A'], [loss_A], epoch)
        self.write_log(callback, ['loss_B'], [loss_B], epoch)
#         self.write_histogram(self.model.autoencoder_A)
#         self.write_histogram(self.model.autoencoder_B)
        
        #root_path = os.path.abspath(os.path.dirname(__file__))
        #path = os.path.join(root_path, "../../logs/test.txt")
        #with open(path, 'a') as log:
        #    log.write("{}\t{}\n".format(loss_A, loss_B))

        if viewer is not None:
            viewer(self.show_sample(target_A[0:14], target_B[0:14]), "training")

    def show_sample(self, test_A, test_B):
        figure_A = numpy.stack([
            test_A,
            self.model.autoencoder_A.predict(test_A),
            self.model.autoencoder_B.predict(test_A),
        ], axis=1)
        figure_B = numpy.stack([
            test_B,
            self.model.autoencoder_B.predict(test_B),
            self.model.autoencoder_A.predict(test_B),
        ], axis=1)

        if test_A.shape[0] % 2 == 1:
            figure_A = numpy.concatenate ([figure_A, numpy.expand_dims(figure_A[0],0) ])
            figure_B = numpy.concatenate ([figure_B, numpy.expand_dims(figure_B[0],0) ])

        figure = numpy.concatenate([figure_A, figure_B], axis=0)
        w = 4
        h = int( figure.shape[0] / w)
        figure = figure.reshape((w, h) + figure.shape[1:])
        figure = stack_images(figure)

        return numpy.clip(figure * 255, 0, 255).astype('uint8')

    
    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
#             summary = tf.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value
#             summary_value.tag = name
            tf.summary.scalar(name, value)
            callback.writer.add_summary(
                tf.summary.scalar(name, value),
                epoch)
            callback.writer.flush()
    
    def write_histogram(self, model):    
        with tf.name_scope(model.name):
            for layer in model.layers:
                w = layer.get_weights()
                print(len(w))
#                 with tf.name_scope(layer.name):
#                     if (len(w) > 0):
#                         print(w)
#                         print("=========================")
#                         weight, bias = layer.get_weights()
#                         tf.summary.histogram('weight', weight)
#                         tf.summary.histogram('bias', bias)
                    