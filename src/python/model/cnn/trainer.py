import os
import time
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from .gcs_utils import download_directory, copy_file_to_gcs
from .dataset import load_dataset, balanced_da_generator, prepare_dataset_for_training
from .model import INPUT_SIZE, ModelBuilder


def main(dataset_bucket, job_dir, learning_rate, batch_size, lr_sched_step,
         da_factor, last_layer_features, nb_epochs, dropout, seed):
    print('Downloading dataset locally')
    dataset_dir = 'dataset2/Train'
    target_dir = job_dir if not job_dir.startswith('gs:/') else '.'
    download_directory('ivadolabs-meetups', dataset_bucket, dataset_dir, target_dir)

    dataset_path = os.path.join(target_dir, dataset_dir)
    category_name_list, category_dict = load_dataset(dataset_path)

    print('Full dataset:')
    for i, v in enumerate(category_name_list):
        print('{: 3d}: {: <30} : {: 3d}'.format(i, category_name_list[i], len(category_dict[category_name_list[i]])))

    # dictionary dataset is loaded at this point
    # need to create train/valid set

    nb_valid_per_category = 50
    train_raw_dict, x_valid, y_valid = prepare_dataset_for_training(category_dict, category_name_list,
                                                                    INPUT_SIZE, nb_valid_per_category, seed)

    print('Valid set:', x_valid.shape, y_valid.shape)
    print('Training set:')
    for i, v in enumerate(category_name_list):
        print('{: 3d}: {: <30} : {: 3d}'.format(i, category_name_list[i], len(train_raw_dict[category_name_list[i]])))

    # training and valid set are ready, let's instanciate the model

    np.random.seed(33)
    mb = ModelBuilder(len(category_name_list), last_layer_features, learning_rate, dropout)
    model = mb.get_model()
    model_name = mb.model_name
    time_str = time.strftime("%Y%m%d-%H%M%S")
    experiment_id = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(time_str, model_name, len(category_name_list),
                                                           last_layer_features, learning_rate, batch_size,
                                                           lr_sched_step, da_factor, dropout, seed)

    # setting up the callbacks

    weight_dir = os.path.join(target_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    weight_path = os.path.join(weight_dir, experiment_id + '-{epoch:03d}-{val_acc:.4f}.h5')
    check_cb = ModelCheckpoint(weight_path, monitor='val_acc', verbose=0, save_best_only=True,
                               save_weights_only=False, period=1, mode='max')

    def step_decay_schedule(initial_lr, decay_factor, step_size):
        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return LearningRateScheduler(schedule)

    lr_sched_cb = step_decay_schedule(learning_rate, 0.75, lr_sched_step)

    log_dir = os.path.join(job_dir, 'logs', experiment_id)
    tf_board_cb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False,
                              write_grads=False, write_images=False)

    # let's train
    model.fit_generator(balanced_da_generator(train_raw_dict, category_name_list, batch_size, INPUT_SIZE, da_factor),
                        steps_per_epoch=5, validation_data=(x_valid, y_valid), verbose=2, epochs=nb_epochs,
                        use_multiprocessing=False, callbacks=[lr_sched_cb, tf_board_cb, check_cb])

    # get the best weight
    weights_list = [f for f in os.listdir(weight_dir) if '.h5' in f]
    weights_list.sort()
    best_weight_path = os.path.join(weight_dir, weights_list[-1])
    best_model = load_model(best_weight_path)

    # sanity check: reevaluate with the best weight
    score = best_model.evaluate(x_valid, y_valid, verbose=0)
    print('Validation results: loss: {:.4f} - accuracy: {:.4f}'.format(score[0], score[1]))

    # can't directly save on GCS because of h5py, so copyit manually
    if job_dir.startswith('gs:/'):
        copy_file_to_gcs(best_weight_path, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--dataset-bucket', type=str, help='Name of the GCS bucket containing the dataset')
    parser.add_argument('--job-dir', type=str, help='Directory containing training artefacts')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr-sched-step', type=int, default=20)
    parser.add_argument('--da-factor', type=float, default=1.2)
    parser.add_argument('--last-layer-features', type=int, default=1024)
    parser.add_argument('--nb-epochs', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=33)
    args = parser.parse_args()

    main(args.dataset_bucket,
         args.job_dir,
         args.learning_rate,
         args.batch_size,
         args.lr_sched_step,
         args.da_factor,
         args.last_layer_features,
         args.nb_epochs,
         args.dropout,
         args.seed)
