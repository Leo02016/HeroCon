import numpy as np
import scipy.io as sio
import os
import itertools
from scipy.io import arff
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import os
import numpy as np
import scipy.io as sio
import random
import cv2
import imutils
import copy
import torch
from sklearn.preprocessing import LabelEncoder


class load_scene():
    def __init__(self, stage, index):
        if stage == 'train' or stage == 'Train':
            if os.path.exists('./data/scene/data_aug_{}.mat'.format(index)):
                data = sio.loadmat('./data/scene/data_aug_{}.mat'.format(index))
                self.data = data['v1_train']
                self.data2 = data['v2_train']
                self.label = data['train_y']
            else:
                train_data = arff.loadarff('./data/scene/scene-train.arff')
                train_df = pd.DataFrame(train_data[0])
                test_data = arff.loadarff('./data/scene/scene-test.arff')
                test_df = pd.DataFrame(test_data[0])
                sample_size = train_df.shape[0] + test_df.shape[0]
                labels = np.zeros((sample_size, 6))
                headers = ["Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban"]
                for i in range(len(headers)):
                    labels[:train_df.shape[0], i] = list(map(int, train_df[headers[i]]))
                    labels[train_df.shape[0]:, i] = list(map(int, test_df[headers[i]]))
                data = np.array(np.concatenate([train_df.to_numpy()[:, :294], test_df.to_numpy()[:, :294]]), dtype=np.float32)
                # even index
                v1 = copy.deepcopy(data)
                v2 = copy.deepcopy(data)
                v1[::2] = v1[::2] + np.random.normal(np.mean(v1[::2]), np.std(v1[::2]), size=v1.shape)[::2] * 0.01
                v2[1::2] = v2[1::2] + np.random.normal(np.mean(v2[1::2]), np.std(v2[1::2]), size=v2.shape)[1::2] * 0.01
                skf = StratifiedKFold(n_splits=20)
                count = 0
                y_new = self.get_new_labels(labels)
                for train_index, test_index in skf.split(data, y_new):
                    if count > 4:
                        break
                    v1_train, v2_train, y_train = v1[train_index], v2[train_index], labels[train_index]
                    v1_test, v2_test, y_test = v1[test_index], v2[test_index], labels[test_index]
                    X_train, X_test = data[train_index], data[test_index]
                    sio.savemat('./data/scene/data_aug_{}.mat'.format(count), {'v1_train': v1_train, 'v2_train': v2_train, 'train_y': y_train,
                                                              'v1_test': v1_test, 'v2_test': v2_test, 'test_y': y_test,
                                                              'train_x': X_train, 'test_x': X_test})
                    count += 1

                data = sio.loadmat('./data/scene/data_aug_{}.mat'.format(index))
                self.data = data['v1_train']
                self.data2 = data['v2_train']
                self.label = data['train_y']
        elif stage == 'test' or stage == 'Test':
            data = sio.loadmat('./data/scene/data_aug_{}.mat'.format(index))
            self.data = data['v1_test']
            self.data2 = data['v2_test']
            self.label = data['test_y']
        else:
            raise(NameError('The stage should be either train or test'))
        self.num_class = self.label[0].shape[0]

    def get_new_labels(self, y):
        y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
        return y_new

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'data2': self.data2[idx], 'label': self.label[idx]}

    def dictionary_search(self, token):
        return self.vocabularies[token]

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.label = self.label[idx]


class load_mnist():
    def __init__(self, stage, num, fold, noise_rate=0):
        self.num_class = 10
        if not os.path.exists('./data/mnist/noisy_mnist_test_{}.mat'.format(noise_rate)):
            import tensorflow as tf
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape((x_train.shape[0], 1, 28, 28)) / 255
            x_test = x_test.reshape((x_test.shape[0], 1, 28, 28)) / 255
            if not os.path.exists('./data/mnist/noisy.mat'):
                noise_train = np.random.normal(np.mean(x_train), np.var(x_train), size=x_train.shape)
                noise_train_v2 = np.random.normal(np.mean(x_train), np.var(x_train), size=x_train.shape)
                noise_test = np.random.normal(np.mean(x_train), np.var(x_train), size=x_test.shape)
                noise_test_v2 = np.random.normal(np.mean(x_train), np.var(x_train), size=x_test.shape)
                sio.savemat('./data/mnist/noisy.mat',
                            {'noise_train': noise_train, 'noise_test': noise_test,
                             'noise_train_v2': noise_train_v2, 'noise_test_v2': noise_test_v2})
            else:
                data = sio.loadmat('./data/mnist/noisy.mat')
                noise_train = data['noise_train']
                noise_test = data['noise_test']
                noise_train_v2 = data['noise_train_v2']
                noise_test_v2 = data['noise_test_v2']
            x_train_v2 = copy.deepcopy(x_train)
            x_test_v2 = copy.deepcopy(x_test)
            noise_train_indices = noise_train < noise_rate
            x_train[noise_train_indices] = x_train[noise_train_indices] + noise_train[noise_train_indices]
            x_train[x_train > 1] = 1
            x_train[x_train < 0] = 0
            x_train = np.array(x_train * 255, dtype=np.int16)
            noise_test_indices = noise_test < noise_rate
            x_test[noise_test_indices] = x_test[noise_test_indices] + noise_test[noise_test_indices]
            x_test[x_test > 1] = 1
            x_test[x_test < 0] = 0
            x_test = np.array(x_test * 255, dtype=np.int16)
            noise_train_v2_indices = noise_train_v2 < noise_rate
            x_train_v2[noise_train_v2_indices] = x_train_v2[noise_train_v2_indices] + noise_train_v2[noise_train_v2_indices]
            x_train_v2[x_train_v2 > 1] = 1
            x_train_v2[x_train_v2 < 0] = 0
            x_train_v2 = np.array(x_train_v2 * 255, dtype=np.int16)
            noise_test_v2_indices = noise_test_v2 < noise_rate
            x_test_v2[noise_test_v2_indices] = x_test_v2[noise_test_v2_indices] + noise_test_v2[noise_test_v2_indices]
            x_test_v2[x_test_v2 > 1] = 1
            x_test_v2[x_test_v2 < 0] = 0
            x_test_v2 = np.array(x_test_v2 * 255, dtype=np.int16)
            sio.savemat('./data/mnist/noisy_mnist_test_{}.mat'.format(noise_rate),
                        {'x_test': np.array(x_train), 'y_test': np.array(y_train)})
            sio.savemat('./data/mnist/noisy_mnist_test_2v_{}.mat'.format(noise_rate),
                        {'v1_test': x_train, 'v2_test': x_train_v2, 'y_test': y_train})
            for i in range(5):
                sio.savemat('./data/mnist/noisy_mnist_train_{}_{}.mat'.format(noise_rate, i),
                            {'x_train': np.array(x_test[num * fold: num * (fold + 1)]).reshape(-1, 1, 28, 28),
                             'y_train': np.array(y_test[num * fold: num * (fold + 1)])})
                sio.savemat('./data/mnist/noisy_mnist_train_2v_{}_{}.mat'.format(noise_rate, i),
                            {'v1_train': x_test[num * fold: num * (fold + 1)].reshape(-1, 1, 28, 28),
                             'v2_train': x_test_v2[num * fold: num * (fold + 1)].reshape(-1, 1, 28, 28),
                             'y_train':  y_test[num * fold: num * (fold + 1)]})
        if stage == 'train' or stage == 'Train':
            data = sio.loadmat('./data/mnist/noisy_mnist_train_2v_{}_{}.mat'.format(noise_rate, fold))
            self.data = data['v1_train']
            self.data2 = data['v2_train']
            self.label = self.label_encoding(data['y_train'][0])
        elif stage == 'test' or stage == 'Test':
            data = sio.loadmat('./data/mnist/noisy_mnist_test_2v_{}.mat'.format(noise_rate))
            self.data = data['v1_test'][:num]
            self.data2 = data['v2_test'][:num]
            self.label = self.label_encoding(data['y_test'][0][:num])
        else:
            raise (NameError('The stage should be either train or test'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'data2': self.data2[idx], 'label': self.label[idx]}

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.label = self.label[idx]

    def one_hot_encoding(self, i, num):
        arr = np.zeros((10))
        count = int(i / num)
        arr[count] = 1.0
        return arr

    def label_encoding(self, label):
        arr = np.zeros((len(label), 10))
        for i in range(len(label)):
            arr[i, label[i]] = 1.0
        return arr


class load_xrmb():
    def __init__(self, stage, num_class, k_fold=5):
        num = int(50000 / num_class)
        if stage == 'train' or stage == 'Train':
            if not os.path.exists('./data/xrmb/xrmb_5_folds_{}.mat'.format(k_fold)):
                data2 = sio.loadmat('./data/XRMBf2KALDI_window7_single2.mat')
                data = sio.loadmat('./data/XRMBf2KALDI_window7_single1.mat')
                index_list = list(map(lambda x: np.where(data2['trainLabel'] == x)[0][:num], range(num_class)))
                indices = np.array(list(itertools.chain.from_iterable(index_list)))
                shuffle = np.random.permutation(len(indices))
                indices = indices[shuffle]
                self.data2 = data2['X2'][indices]
                self.num_class = num_class
                self.label = self.label_encoding(data2['trainLabel'][indices])
                self.data = data['X1'][indices]
                skf = StratifiedKFold(n_splits=200)
                count = 0
                for train_index, test_index in skf.split(self.data, data2['trainLabel'][indices]):
                    v1_train, v2_train, y_train = self.data[test_index], self.data2[test_index], self.label[test_index]
                    v1_test, v2_test, y_test = self.data[train_index], self.data2[train_index], self.label[train_index]
                    sio.savemat('./data/xrmb/xrmb_5_folds_{}.mat'.format(count),
                                {'train_v1': v1_train, 'train_v2': v2_train, 'train_y': y_train,
                                 'test_v1': v1_test, 'test_v2': v2_test, 'test_y': y_test})
                    count += 1
                    if count >= 5:
                        break
                data = sio.loadmat('./data/xrmb/xrmb_5_folds_0.mat')
                self.data = data['train_v1']
                self.data2 = data['train_v2']
                self.label = data['train_y']
            else:
                data = sio.loadmat('./data/xrmb/xrmb_5_folds_{}.mat'.format(k_fold))
                self.num_class = num_class
                self.data = data['train_v1']
                self.data2 = data['train_v2']
                self.label = data['train_y']
        elif stage == 'test' or stage == 'Test':
            data = sio.loadmat('./data/xrmb/xrmb_5_folds_{}.mat'.format(k_fold))
            self.num_class = num_class
            self.data = data['test_v1']
            self.data2 = data['test_v2']
            self.label = data['test_y']
        else:
            raise (NameError('The stage should be either train or test'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        v2 = self.data2[idx]
        y = self.label[idx]
        sample = {'data': np.array(data), 'data2': np.array(v2), 'label': y}
        return sample

    def label_encoding(self, label):
        arr = np.zeros((len(label), self.num_class))
        for i in range(len(label)):
            arr[i, label[i]] = 1.0
        return arr

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.label = self.label[idx]

    def cosine_distance_torch(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.exp(torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature))


class load_celeba():
    def __init__(self, stage,  i):
        data_path = './data/'
        # data_path = './dataset/img_align_celeba/'
        file_list = ['./data/celeba/celeba_5_folds_{}.mat'.format(i)]
        aaa = list(map(os.path.exists, file_list))
        self.num_class = 40
        if sum(aaa) != len(aaa):
            self.preprocess(data_path, 5)
        if stage == 'test' or stage == 'Test':
            data = sio.loadmat('{}/celeba/celeba_5_folds_{}.mat'.format(data_path, i))
            self.data = np.array(data['test_v1'])
            self.data2 = np.array(data['test_v2'])
            # label = sio.loadmat('{}/celeba_label.mat'.format(data_path))
            self.label = self.label_encoding(data['test_y'])
            # self.num_class = self.label.shape[1]
        elif stage == 'train' or stage == 'Train':
            data = sio.loadmat('{}/celeba/celeba_5_folds_{}.mat'.format(data_path, i))
            self.data = np.array(data['train_v1'])
            self.data2 = np.array(data['train_v2'])
            self.label = self.label_encoding(data['train_y'])
        else:
            raise (NameError('The stage should be either train or test'))

    def label_encoding(self, label):
        arr = np.zeros((len(label), self.num_class))
        arr[np.where(label == 1.0)] = 1.0
        return arr

    def preprocess(self, data_path, k_fold):
        v1 = []
        v2 = []
        labels = []
        label_index = list(range(1, self.num_class + 1))
        count = 1
        with open('./data/list_attr_celeba.txt', 'r+') as f:
            next(f)
            next(f)
            for line in f.readlines():
                data = np.array(line.split())
                label = list(map(int, data[label_index]))
                # crop and resize the image to generate the first view
                # img = cv2.imread(data_path + 'img_align_celeba/' + data[0])
                # assert img is not None, 'File {} is not loaded correctly'.format(data[0])
                # x, y, h, w = 20, 0, 160, 200
                # crop_img = img[x:x+h, y:y+w, :]
                # cv2.imwrite(data_path + 'img_align_celeba_v1/' + data[0], crop_img)
                v1.append(data_path + 'img_align_celeba_v1/' + data[0])
                # color distortion to generate the second view
                # https://stackoverflow.com/questions/35152636/random-flipping-and-rgb-jittering-slight-value-change-of-image
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # h, w, c = img.shape
                # noise = np.random.randint(0, 50, (h, w))
                # zitter = np.zeros_like(img)
                # zitter[:, :, 1] = noise
                # noise_added = cv2.add(img, zitter)
                # combined_img = np.vstack((img[:int(h / 2), :, :], noise_added[int(h / 2):, :, :]))
                # cv2.imwrite(data_path + 'img_align_celeba_v2/' + data[0], combined_img)
                v2.append(data_path + 'img_align_celeba_v2/' + data[0])
                labels.append(label)
                if count >= 50000:
                    break
                else:
                    count += 1
        v1 = np.array(v1)
        v2 = np.array(v2)
        labels = np.array(labels)
        # train_index = range(180000)
        # test_index = range(180000, len(v1))
        # train_v1 = v1[train_index]
        # test_v1 = v1[test_index]
        # train_v2 = v2[train_index]
        # test_v2 = v2[test_index]
        # train_label = labels[train_index]
        # test_label = labels[test_index]
        # train_x = []
        # for image in train_image:
        #     train_x.append(self.load_image(image, 224))
        # test_x = []
        # for image in test_image:
        #     test_x.append(self.load_image(image, 224))
        # sio.savemat('{}/data.mat'.format(data_path), mdict={'train': train_x, 'test': test_x})
        # sio.savemat('{}/label.mat'.format(data_path), mdict={'train': train_label, 'test': test_label})
        # sio.savemat('{}/celeba_data.mat'.format(data_path), mdict={'train': train_image, 'test': test_image})
        # sio.savemat('{}/celeba_label.mat'.format(data_path), mdict={'train': train_label, 'test': test_label})
        # skf = StratifiedKFold(n_splits=100)
        # count = 0
        # y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in labels])
        # for train_index, test_index in skf.split(v1, y_new):
        for i in range(5):
            indices = np.random.permutation(len(v1))
            train_index = indices[:500]
            test_index = indices[500:]
            v1_train, v2_train, y_train = v1[train_index], v2[train_index], labels[train_index]
            v1_test, v2_test, y_test = v1[test_index], v2[test_index], labels[test_index]
            sio.savemat('./data/celeba/celeba_5_folds_{}.mat'.format(i),
                        {'train_v1': v1_train, 'train_v2': v2_train, 'train_y': y_train,
                         'test_v1': v1_test, 'test_v2': v2_test, 'test_y': y_test})

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def load_image(self, image_name, scale):
        blob = np.zeros((scale, scale, 3), dtype=np.float32)
        image_name = image_name[4:]
        imgs = cv2.imread(image_name)
        assert imgs is not None, 'File {} is not loaded correctly'.format(image_name)
        if imgs.shape[0] > scale or imgs.shape[1] > scale:
            pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
            imgs = self.prep_im_for_blob(imgs, pixel_means, scale)
        blob[0:imgs.shape[0], 0:imgs.shape[1], :] = imgs
        channel_swap = (2, 0, 1)
        blob = blob.transpose(channel_swap)
        return blob

    def prep_im_for_blob(self, im, pixel_means, scale):
        """ Mean subtract and scale an image """
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im = cv2.resize(im, dsize=(scale, scale), interpolation=cv2.INTER_LINEAR)
        return im

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = []
        for image in self.data[idx]:
            data.append(self.load_image(image, 224))
        data = np.array(data)
        data2 = []
        for image in self.data2[idx]:
            data2.append(self.load_image(image, 224))
        data2 = np.array(data2)
        return {'data': data, 'data2': data2, 'label': self.label[idx]}

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.label = self.label[idx]