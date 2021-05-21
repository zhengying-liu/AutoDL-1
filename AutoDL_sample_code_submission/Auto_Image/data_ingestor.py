from automl_workflow.api import Dataset, DataIngestor

from Auto_Image.skeleton.data.dataset import TFDataset


class MyDataIngestor(DataIngestor):

    def ingest(self, dataset):
        """
        Args:
          dataset: tf.data.Dataset object

        Returns:
          a (pytorch) Dataset object.
        """
        return super().ingest(dataset)