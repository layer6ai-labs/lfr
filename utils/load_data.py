import h5py
import torch
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import SubsetRandomSampler as Sampler

from utils.data_transforms import get_transforms

def get_data_path(dataset):
    _DATASET_PATHS = {'uci-income': 'uci-income',
                      'theorem': 'theorem',
                      'har': 'HAR',
                      'epilepsy': 'EPILEPSY',
                      'hepmass': 'hepmass',
                      'kvasir': 'kvasir',
                      'mimic3-los': 'mimic3',
                      }
    data_path = _DATASET_PATHS.get(dataset, "")
    return 'data/' + data_path


def load_data(args, dataset=None, labelled=False, train_sample_frac: float = 1.0):
    data_path = get_data_path(dataset)
    if dataset in ['uci-income', 'theorem', 'epilepsy', 'hepmass']:
        train_loader, test_loader = load_from_numpy(args, data_path, labelled=labelled, train_sample_frac=train_sample_frac)
    elif dataset == 'har':
        train_loader, test_loader = load_from_pt(args, data_path, labelled=labelled, train_sample_frac=train_sample_frac)
    elif dataset == 'kvasir':
        train_loader, test_loader = load_from_numpy(args, data_path, labelled=labelled, train_sample_frac=train_sample_frac)
    elif dataset == 'mimic3-los':
        train_loader, test_loader = load_mimic3(args, data_path, task='los', labelled=labelled, train_sample_frac=train_sample_frac)
    else:
        raise ValueError(f'Unknown dataset {dataset}')
    
    return train_loader, test_loader


def create_loaders(args, X_train, y_train, X_test, y_test, labelled=False, train_sample_frac: float = 1.0):
    transforms_train, transforms_test = get_transforms(args.method, args.dataset, labelled)
    if args.dataset in ['theorem', 'uci-income', 'hepmass']:
        train_dataset = TabularDataset(X_train, y_train, transform=transforms_train)
        test_dataset = TabularDataset(X_test, y_test, transform=transforms_test)
    else:
        train_dataset = Dataset(X_train, y_train, transform=transforms_train)
        test_dataset = Dataset(X_test, y_test, transform=transforms_test)
    
    if 'diet' in args.method and not labelled:
        train_dataset = DatasetWithIndices(train_dataset)
        test_dataset = DatasetWithIndices(test_dataset)
    if train_sample_frac < 1.0:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                   sampler=Sampler(get_sampled_indices(train_dataset, train_sample_frac)),
                                                   num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader


def load_from_numpy(args, path, labelled=False, train_sample_frac: float = 1.0):
    X_train = torch.from_numpy(np.load(f'{path}/X_train.npy'))
    X_test = torch.from_numpy(np.load(f'{path}/X_test.npy'))
    y_train = torch.from_numpy(np.load(f'{path}/y_train.npy'))
    y_test = torch.from_numpy(np.load(f'{path}/y_test.npy'))
    
    return create_loaders(args, X_train, y_train, X_test, y_test, labelled=labelled, 
                          train_sample_frac=train_sample_frac)


def load_from_pt(args, data_path, labelled=False, train_sample_frac: float = 1.0):
    train_data = torch.load(f'{data_path}/train.pt')
    X_train, y_train = train_data['samples'], train_data['labels']
    assert len(X_train.size()) == 3
    test_data = torch.load(f'{data_path}/test.pt')
    X_test, y_test = test_data['samples'], test_data['labels']

    return create_loaders(args, X_train, y_train, X_test, y_test, labelled=labelled, 
                          train_sample_frac=train_sample_frac)


def load_kvasir(args, data_path, labelled=False, train_sample_frac: float = 1.0):
    data_file = f"{data_path}.npz"
    dataset = np.load(data_file)

    # 8000 into (6000 train and 2000 test)
    assert dataset['images'].shape[0] == 8000
    train_idx = []
    test_idx = []
    for c in range(8):  # class-stratified partitioning
        cls_idx = np.where(dataset['labels'] == c)[0]
        np.random.shuffle(cls_idx)
        train_idx.append(cls_idx[:750])
        test_idx.append(cls_idx[750:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    X_train = dataset['images'][train_idx].transpose([0, 3, 1, 2]) / 255
    y_train = dataset['labels'][train_idx].astype(np.int64)
    X_test = dataset['images'][test_idx].transpose([0, 3, 1, 2]) / 255
    y_test = dataset['labels'][test_idx].astype(np.int64)
    
    return create_loaders(args, X_train, y_train, X_test, y_test, labelled=labelled, 
                          train_sample_frac=train_sample_frac)


def load_center(data_path, center, split):
    image_data_file = f"{data_path}/images_{split}_{center}.npy"
    meta_data_file = f"{data_path}/labels_centers_patients_{split}_{center}.npy"
    image_dataset = np.load(image_data_file)
    meta_dataset = np.load(meta_data_file)
    assert image_dataset.shape[0] == meta_dataset.shape[0]

    return image_dataset, meta_dataset[:, 0]

def load_mimic3(args, data_path, task, labelled=False, train_sample_frac: float = 1.0):
    SPLITS = ['train', 'test']
    WINDOW_SIZE = 48
    data_h5 = h5py.File(data_path, "r")
    # load task data
    labels_name = data_h5['labels'].attrs['tasks']
    label_idx = np.where(labels_name == task)[0]
    datasets = {}

    transforms = get_transforms(args.method, args.dataset, labelled)
    with_indices = False
    if 'diet' in args.method and not labelled:
        with_indices = True
    for i, split in enumerate(SPLITS):
        data = data_h5['data'][split][:, :]    # shape: (sample_num, attributes)
        labels = np.reshape(data_h5['labels'][split][:, label_idx], (-1,))   # shape: (sample_num)
        patient_windows= data_h5['patient_windows'][split][:]   # shape: (patient_num, (start_window, end_window, patient_id))
        datasets[split] = MIMIC3Dataset(data, labels, patient_windows, task, WINDOW_SIZE, 
                                        load_labelled_only=labelled, with_indices=with_indices, transform=transforms[i])

    if train_sample_frac < 1.0:
        train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size, 
                                                   sampler=Sampler(get_sampled_indices(datasets['train'], train_sample_frac)),
                                                   num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
        train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size=args.eval_bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader


def get_sampled_indices(dataset, sample_frac):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    sampled_indices = np.random.choice(indices, int(np.floor(sample_frac * dataset_size)))
    print(f'sampling {100 * sample_frac}% of data, which is {len(sampled_indices)} of {dataset_size}.')
    return sampled_indices


class MIMIC3Dataset(torch.utils.data.Dataset):
    'MIMIC3 dataset'
    def __init__(self, features: np.ndarray, labels: np.ndarray, patient_windows: np.ndarray, 
                 task: str, window_size: int=48, load_labelled_only: bool=True, with_indices: bool=False, transform=None):
        self.features = features
        self.labels = labels
        self.patient_windows = patient_windows
        self.window_size = window_size
        self.load_labelled_only = load_labelled_only
        self.with_indices = with_indices
        self.task = task
        self.bins = [24, 48, 72, 96, 120, 144, 168, 192, 336]
        self.transform = transform

        self.preprocess()

    def __len__(self):
        'Denotes the total number of samples'
        if self.load_labelled_only:
            return len(self.valid_labels_indexes)
        else:
            return len(self.samples_indexes)

    def __getitem__(self, index: int):
        'Generates one sample of data'
        # transforming the index into the index of labeled data
        if self.load_labelled_only:
            index = self.valid_labels_indexes[index]
        
        relative_idx = self.samples_relative_indexes[index]
        # Load data and get label
        X = self.get_windowed_seq(index, relative_idx)
        y = self.labels[index]

        if self.transform:
            X = self.transform(torch.tensor(X))
        
        if self.with_indices:
            return index, X
        else:
            return X, y
    
    def preprocess(self):
        'Bins the labels and filters the labeled data and get relative indexes for sequence windowing'
        if self.task == 'los':  # this turns the labels into discrete values, so as to form a classification task
            self.labels = torch.bucketize(torch.tensor(self.labels), torch.tensor(self.bins), right=True).numpy()

        if self.load_labelled_only:
            self.valid_labels_indexes = np.argwhere(~np.isnan(self.labels)).reshape(-1) # indexes of samples with labels, shape: (sample_num)

        self.samples_indexes = []
        self.samples_relative_indexes = []
        self.indexes_to_episodes = []

        for episode_id, (start, stop, patient_id) in enumerate(self.patient_windows):
            # get all samples indexes for x
            samples_indexes = np.arange(start, stop + 1)
            self.samples_indexes += list(samples_indexes)
            # get relative indexes for x
            samples_relative_indexes = np.arange(0, stop + 1 - start)
            self.samples_relative_indexes += list(samples_relative_indexes)
            # mark the episodes for x
            self.indexes_to_episodes = list(np.ones((stop + 1 - start,)) * episode_id)
        self.samples_indexes = np.array(self.samples_indexes)
        self.samples_relative_indexes = np.array(self.samples_relative_indexes)
        self.indexes_to_episodes = np.array(self.indexes_to_episodes)
        
    
    def get_windowed_seq(self, index: int, relative_idx: int):
        'Returns a input sequence with the specified window size'
        seq_len = relative_idx + 1
        if seq_len < self.window_size:
            seq = self.features[index - relative_idx : index + 1]
            padding = np.zeros((self.window_size - relative_idx - 1, self.features.shape[-1]))
            seq = np.concatenate((padding, seq), axis=0)
        else:
            seq = self.features[index + 1 - self.window_size : index + 1]
        
        return seq


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.features = features
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        feature = self.features[index]
        label = self.labels[index]
        if self.transform:
            feature = self.transform(feature)
        return feature, label


class TabularDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.features = features
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        feature = self.features[index]
        label = self.labels[index]
        if self.transform:
            random_idx = np.random.randint(0, len(self))
            random_sample = self.features[random_idx]
            feature = self.transform(feature, random_sample)
        return feature, label


class DatasetWithIndices(torch.utils.data.Dataset):
    'Custom a dataset with transform'
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]            
