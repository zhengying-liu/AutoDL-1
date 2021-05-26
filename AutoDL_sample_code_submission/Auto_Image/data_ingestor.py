from automl_workflow.api import Dataset, DataIngestor

from Auto_Image.skeleton.data.dataset import TFDataset


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
        dataset = skeleton.data.TFDataset(self.session, dataset, num_items=50)
        return dataset