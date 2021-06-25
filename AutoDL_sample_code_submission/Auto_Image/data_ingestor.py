from automl_workflow.api import Dataset, DataIngestor
from Auto_Image.skeleton.data.dataset import TFDataset
import tensorflow as tf
import Auto_Image.skeleton as skeleton
from Auto_Image.skeleton.projects.others import *

class MyDataIngestor(DataIngestor):

    def __init__(self, session=None, info=None):
        super.__init__()
        self.session = session
        self.info = info

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


class MyDataIngestor2(DataIngestor):

    def __init__(self, session=None, info=None):
        self.session = session
        self.info = info
        self.metadata = self.info['metadata']
        print('-' * 30)
        print(self.info)

    def ingest(self, dataset, mode='train'):
        """
        Args:
        dataset: tf.data.Dataset object
        Returns:
        a (pytorch) Dataset object.
        """
        self.prepare_hyperparams() # get self.info and self.hyperparams

        if not self.info['condition']['first']['train']:
            return self.build_or_get_dataloader('train')

        num_images = self.info['dataset']['size']

        num_valids = int(min(num_images * self.hyper_params['dataset']['cv_valid_ratio'],
                                self.hyper_params['dataset']['max_valid_count']))
        num_trains = num_images - num_valids

        num_samples = self.hyper_params['dataset']['train_info_sample']
        sample = dataset.take(num_samples).prefetch(buffer_size=num_samples)
        train = skeleton.data.TFDataset(self.session, sample, num_samples)
        
        self.info['dataset']['sample'] = train.scan(samples=num_samples)
        del train
        del sample

        times, height, width, channels = self.info['dataset']['sample']['example']['shape']
        values = self.info['dataset']['sample']['example']['value']
        aspect_ratio = width / height

        if aspect_ratio > 2 or 1. / aspect_ratio > 2:
            self.hyper_params['dataset']['max_size'] *= 2
            self.need_switch = False
        size = [min(s, self.hyper_params['dataset']['max_size']) for s in [height, width]]


        if aspect_ratio > 1:
            size[0] = size[1] / aspect_ratio
        else:
            size[1] = size[0] * aspect_ratio

        if width <= 32 and height <= 32:
            input_shape = [times, height, width, channels]
        else:
            fit_size_fn = lambda x: int(x / self.hyper_params['dataset']['base'] + 0.8) * self.hyper_params['dataset'][
                'base']
            size = list(map(fit_size_fn, size))
            min_times = min(times, self.hyper_params['dataset']['max_times'])
            input_shape = [fit_size_fn(min_times) if min_times > self.hyper_params['dataset'][
                'base'] else min_times] + size + [channels]
        self.input_shape = input_shape


        self.hyper_params['dataset']['input'] = input_shape

        num_class = self.info['dataset']['num_class']
        batch_size = self.hyper_params['dataset']['batch_size']
        step_num = self.hyper_params['dataset']['steps_per_epoch']
        if num_class > batch_size / 2:
            self.hyper_params['dataset']['batch_size'] = batch_size * 2
        preprocessor1 = get_tf_resize(input_shape[1], input_shape[2], times=input_shape[0], min_value=values['min'],
                                        max_value=values['max'])


        dataset = dataset.map(
            lambda *x: (preprocessor1(x[0]), x[1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        must_shuffle = self.info['dataset']['sample']['label']['zero_count'] / self.info['dataset']['num_class'] >= 0.5
        enough_count = self.hyper_params['dataset']['enough_count']['image']
        if must_shuffle or num_images < enough_count:
            dataset = dataset.shuffle(buffer_size=4 * num_valids, reshuffle_each_iteration=False)

        train = dataset.skip(num_valids)
        valid = dataset.take(num_valids)
        self.datasets = {
            'train': train,
            'valid': valid,
            'num_trains': num_trains,
            'num_valids': num_valids
        }

        # build_or_get_dataloader

        num_items = num_trains

        if mode in self.dataloaders and self.dataloaders[mode] is not None:
            return self.dataloaders[mode]

        enough_count = self.hyper_params['dataset']['enough_count']['image']

        values = self.info['dataset']['sample']['example']['value']
        if mode == 'train':
            batch_size = self.hyper_params['dataset']['batch_size']
            preprocessor = get_tf_to_tensor(is_random_flip=True)

            if num_items < enough_count:
                dataset = dataset.cache()

            dataset = dataset.repeat()
            dataset = dataset.map(
                lambda *x: (preprocessor(x[0]), x[1]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            dataset = dataset.prefetch(buffer_size=batch_size * 8)

            dataset = skeleton.data.TFDataset(self.session, dataset, num_items)
            transform = tv.transforms.Compose([
            ])
            dataset = skeleton.data.TransformDataset(dataset, transform, index=0)


        print("dataset with size: " + str(len(dataset)))
        for i in range(len(dataset)): 
            print(dataset[i][0].shape, dataset[i][1].shape) # Expected: (3, 32, 32) (10,)
        return dataset




    def prepare_hyperparams(self):
        self.info = {
            'dataset': {
                'path': self.metadata.get_dataset_name(),
                'shape': self.metadata.get_tensor_size(0),
                'size': self.metadata.size(),
                'num_class': self.metadata.get_output_size()
            },
            'loop': {
                'epoch': 0,
                'test': 0,
                'best_score': 0.0
            },
            'condition': {
                'first': {
                    'train': True,
                    'valid': True,
                    'test': True
                }
            },
            'terminate': False
        }

        self.hyper_params = {
            'optimizer': {
                'lr': 0.05,
            },
            'dataset': {
                'train_info_sample': 256,
                'cv_valid_ratio': 0.1,
                'max_valid_count': 256,

                'max_size': 64,
                'base': 16,  #
                'max_times': 8,

                'enough_count': {
                    'image': 10000,
                },

                'batch_size': 32,
                'steps_per_epoch': 30,
                'max_epoch': 1000,  #
                'batch_size_test': 64,
            },
            'checkpoints': {
                'keep': 50
            },
            'conditions': {
                'score_type': 'auc',
                'early_epoch': 1,
                'skip_valid_score_threshold': 0.90,  #
                'skip_valid_after_test': min(10, max(3, int(self.info['dataset']['size'] // 1000))),
                'test_after_at_least_seconds': 1,
                'test_after_at_least_seconds_max': 90,
                'test_after_at_least_seconds_step': 2,

                'threshold_valid_score_diff': 0.001,
                'threshold_valid_best_score': 0.997,
                'max_inner_loop_ratio': 0.2,
                'min_lr': 1e-6,
                'use_fast_auto_aug': True
            }
        }
        self.dataloaders = {
            'train': None,
            'valid': None,
            'test': None
        }
    
