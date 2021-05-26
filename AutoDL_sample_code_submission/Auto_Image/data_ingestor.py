from automl_workflow.api import Dataset, DataIngestor

from Auto_Image.skeleton.data.dataset import TFDataset

import tensorflow as tf


class MyDataIngestor(DataIngestor):

    def __init__(self, session):
      self.session = session

    def ingest(self, dataset):
        """
        Args:
          dataset: tf.data.Dataset object

        Returns:
          a (pytorch) Dataset object.
        """
        num_samples = 0

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
          while True:
            try:
              _ = sess.run(next_element)
              num_samples += 1
            except tf.errors.OutOfRangeError:
              break

        pt_dataset = TFDataset(None, dataset, num_samples)

        return pt_dataset
