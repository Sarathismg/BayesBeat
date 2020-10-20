import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader

NUM_WORKERS = 0

class AFdataset(Dataset):
    def __init__(self, ds_npz):
        super().__init__()
        self.ds_npz = ds_npz

    def __getitem__(self, index: int):
        X = self.ds_npz['signal'][index]
        rhythm = self.ds_npz['rhythm'][index]
        qa = self.ds_npz['qa_label'][index]
        X, qa, rhythm = np.array(X, dtype='float32').reshape((1, 800)), qa, rhythm.astype('float32')
        return X, qa, rhythm

    def __len__(self):
        return len(self.ds_npz['signal'])


def getGenerator(npz_file, indices_array, total_batch_size, replacement=False, is_train=True, shuffle=True, remove_poor = False):
    dataset = AFdataset(npz_file)
    weights = np.zeros(len(npz_file['qa_label']))

    # [-4-],[-4-],[-2-] -> [-0.25-],[-0.25],[-0.5-] independent-coverage?
    for indices in indices_array:
        if remove_poor:
            weights[indices] = 1
        else:
            weights[indices] = (1 / len(indices_array)) * (1 / len(indices))

    # create the sampler
    if is_train or remove_poor:
        batchsampler = WeightedRandomSampler(weights=weights, num_samples=int(np.sum([len(k) for k in indices_array])),
                                             replacement=replacement)
        dataloader = DataLoader(dataset, batch_size=total_batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                sampler=batchsampler, pin_memory=True)
        return dataloader

    return DataLoader(dataset, batch_size=total_batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)


def get_weighted_generator(file_location, batch_size, replacement, is_train=True, shuffle=True, percentage=1.0, remove_poor = False):
    #signal = np.load(file_location + '/signal.npy', mmap_mode='r')
    signal = np.load(file_location + '/signal.npy')
    qa = np.load(file_location + '/qa_label.npy')
    rhythm = np.load(file_location + '/rhythm.npy')
    tr = {}
    tr['signal'] = signal
    tr['qa_label'] = qa
    tr['rhythm'] = rhythm

    # get array of array of indices each array represents one cat
    indices = []
    if not remove_poor:
        indices.append(np.where((tr['qa_label'][:, 0] == 1) & (tr['rhythm'][:, 0] == 1))[0])  # poor, non-af
        indices.append(np.where((tr['qa_label'][:, 0] == 1) & (tr['rhythm'][:, 1] == 1))[0])  # poor, af
    indices.append(np.where((tr['qa_label'][:, 1] == 1) & (tr['rhythm'][:, 0] == 1))[0])  # okay, non-af
    indices.append(np.where((tr['qa_label'][:, 1] == 1) & (tr['rhythm'][:, 1] == 1))[0])  # okay, af
    indices.append(np.where((tr['qa_label'][:, 2] == 1) & (tr['rhythm'][:, 0] == 1))[0])  # excellent, non-af
    indices.append(np.where((tr['qa_label'][:, 2] == 1) & (tr['rhythm'][:, 1] == 1))[0])  # excellent, af

    for k in range(len(indices)):
        np.random.shuffle(indices[k])
        indices[k] = indices[k][0: int(len(indices[k]) * percentage)]

    return getGenerator(tr, indices, batch_size, replacement=replacement, is_train=is_train, shuffle=shuffle, remove_poor=remove_poor)
