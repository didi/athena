# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Only support eager mode
# pylint: disable=invalid-name
r""" check point manager """
import os
import tensorflow as tf
from absl import logging
import numpy as np
import shutil


class Checkpoint(tf.train.CheckpointManager):
    """ A wrapper for Tensorflow checkpoint

    Args:
        checkpoint_directory: the directory for checkpoint
        summary_directory: the directory for summary used in Tensorboard
        __init__ provide the optimizer and model
        __call__ save the model

    Example:
        transformer = SpeechTransformer(target_vocab_size=dataset_builder.target_dim)
        optimizer = tf.keras.optimizers.Adam()
        ckpt = Checkpoint(checkpoint_directory='./train', summary_directory='./event',
            model=transformer, optimizer=optimizer)
        solver = BaseSolver(transformer)
        for epoch in dataset:
            loss = solver.train(..)
            ckpt(loss)
    """
    def __init__(self, checkpoint_directory=None, **kwargs):
        self.ckpt = tf.train.Checkpoint(**kwargs)
        if checkpoint_directory is None:
            checkpoint_directory = os.path.join(os.path.expanduser("~"), ".athena")
        super().__init__(self.ckpt, directory=checkpoint_directory, max_to_keep=5)
        self.best_loss = np.inf
        self.checkpoint_directory = checkpoint_directory
        self.save_counter = self.ckpt.save_counter
        logging.info("trying to restore from : %s" % checkpoint_directory)
        # load from latest checkpoint if previous models exist in checkpoint dir
        self.ckpt.restore(tf.train.latest_checkpoint(checkpoint_directory))
        self.best_checkpoint_directory = os.path.join(self.checkpoint_directory, 'best_loss')

    def _compare_and_save_best(self, loss, save_path):
        """ compare and save the best model in best_loss """
        if loss is None:
            return
        if loss < self.best_loss:
            self.best_loss = loss
            # save ckpt of best loss
            best_dir = tf.train.latest_checkpoint(self.checkpoint_directory)
            best_name = os.path.basename(best_dir)
            self.copy_best_ckpt(best_name)

    def __call__(self, loss=None):
        logging.info("saving model in :%s" % self.checkpoint_directory)
        save_path = self.save()
        self._compare_and_save_best(loss, save_path)

    def restore_from_best(self):
        """ restore from the best model """
        self.ckpt.restore(
            tf.train.latest_checkpoint(
                self.checkpoint_directory,
                latest_filename='best_loss'
            )
        )

    def copy_best_ckpt(self, best_name):
        if not os.path.exists(self.best_checkpoint_directory):
            os.makedirs(self.best_checkpoint_directory)
        files = ['.data-00000-of-00002', '.data-00001-of-00002', '.index']
        for i in files:
            src = self.checkpoint_directory + best_name + i
            dst = self.best_checkpoint_directory + '/ckpt-1' + i
            shutil.copyfile(src, dst)

