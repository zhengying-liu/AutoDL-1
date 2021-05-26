import Auto_Image.skeleton


# Need to revise according to api
# the correct version are in data_ingestor.py

class MyDataIngestor(self):
    def ingest(self, session, dataset, num_samples, input_shape, batch_size, steps_per_epoch):
        # copied from logic.py, build_or_get_train_dataloader()
        # preprocessor1 = get_tf_resize(input_shape[1], input_shape[2], times=input_shape[0], min_value=values['min'], max_value=values['max'])

        preprocessor1 = get_tf_resize(input_shape[1], input_shape[2], input_shape[0])        
        dataset = dataset.map(
            lambda *x: (preprocessor1(x[0]), x[1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        preprocessor = get_tf_to_tensor(is_random_flip=True)
        dataset = dataset.repeat()
        dataset = dataset.map(
            lambda *x: (preprocessor(x[0]), x[1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = dataset.prefetch(buffer_size=batch_size * 8)

        dataset = skeleton.data.TFDataset(self.session, dataset, num_items)

        train_dataloader = skeleton.data.FixedSizeDataLoader(
            dataset,
            # steps=self.hyper_params['dataset']['steps_per_epoch'],
            steps=steps_per_epoch,
            batch_size=batch_size,
            shuffle=False, drop_last=True, num_workers=0, pin_memory=False
        )

        return train_dataloader



"""
    Auto_Image.skeleton.projects.api.others.py
"""
def get_tf_resize(height, width, times=1, min_value=0.0, max_value=1.0):
    def preprocessor(tensor):
        in_times, in_height, in_width, in_channels = tensor.get_shape()
        LOGGER.info('[get_tf_resize] shape:%s', (in_times, in_height, in_width, in_channels))

        if width == in_width and height == in_height:
            LOGGER.info('[get_tf_resize] do not resize (%dx%d)', width, height)
        else:
            tensor = tf.image.resize_images(tensor, (height, width), method=tf.image.ResizeMethod.BICUBIC)

        if times != in_times or times > 1:
            # resize time axis using NN (to select frame \wo interpolate)
            tensor = tf.reshape(tensor, [-1, height * width, in_channels])
            tensor = tf.image.resize_images(tensor, (times, height * width),
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            tensor = tf.reshape(tensor, [times, height, width, in_channels])

        if times == 1:
            tensor = tensor[int(times // 2)]

        delta = max_value - min_value
        if delta < 0.9 or delta > 1.1 or min_value < -0.1 or min_value > 0.1:
            LOGGER.info('[get_tf_resize] min-max normalize(min:%f, max:%f)', min_value, max_value)
            tensor = (tensor - min_value) / delta

        return tensor

    return preprocessor


"""
    Auto_Image.skeleton.projects.api.others.py
"""
def get_tf_to_tensor(is_random_flip=True):
    def preprocessor(tensor):
        if is_random_flip:
            tensor = tf.image.random_flip_left_right(tensor)
            # tensor = tf.image.random_flip_up_down(tensor)
            # if random.random() > 0.5:
            #     tensor = tf.image.rot90(tensor, k=1)
        dims = len(tensor.shape)
        # LOGGER.info('[get_tf_to_tensor] dims:%s', dims)
        if dims == 3:
            # height, width, channels -> channels, height, width
            tensor = tf.transpose(tensor, perm=[2, 0, 1])
        elif dims == 4:
            # time, height, width, channels -> time, channels, height, width
            tensor = tf.transpose(tensor, perm=[0, 3, 1, 2])
        return tensor

    return preprocessor