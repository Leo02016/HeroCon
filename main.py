from data_loader import *
import time
from module import *
import torch
from sklearn.metrics import f1_score, roc_auc_score
import argparse
import matplotlib
matplotlib.use('Agg')
from LARS_optimizer import LARC


def main(args):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu>=0 else 'cpu')
    if args.data == 'scene':
        test_dataset = load_scene('train', args.fold)
        train_dataset = load_scene('test', args.fold)
        batch_size = 20000
        batch_size_unlabel = 20000
        d_input = train_dataset.data.shape[1]
        d_2 = train_dataset.data2.shape[1]
        multi_label = True
    elif args.data == 'xrmb':
        train_dataset = load_xrmb('train', 20, args.fold)
        test_dataset = load_xrmb('test', 20, args.fold)
        batch_size = 250
        batch_size_unlabel = 250
        d_input = train_dataset.data.shape[1]
        d_2 = train_dataset.data2.shape[1]
        multi_label = False
    elif args.data == 'celeba':
        train_dataset = load_celeba('train', args.fold)
        test_dataset = load_celeba('test', args.fold)
        batch_size = 32 * 2
        batch_size_unlabel = 32 * 2
        d_input = 100
        d_2 = d_input
        multi_label = True
    elif args.data == 'mnist':
        train_dataset = load_mnist('train',  200, args.fold, noise_rate=0.1)
        test_dataset = load_mnist('test', 60000, args.fold, noise_rate=0.1)
        batch_size = 200
        batch_size_unlabel = 1000
        d_input = 784
        d_2 = 784
        multi_label = False
    parallel_mode = False
    if args.model == 'vgg':
        model = vggcpc(train_dataset.num_class, in_channels=3, alpha=args.alpha, beta=args.beta,
                       gpu=args.gpu).to(device)
        parallel_mode = True
    elif args.model == 'cnn':
        model = cnncpc(train_dataset.num_class, d_feature=d_input, in_channel=1, alpha=args.alpha, beta=args.beta,
                       gpu=args.gpu).to(device)
    elif args.model == 'linear':
        model = mlcpc(train_dataset.num_class, d_feature=d_input, d_2=d_2, alpha=args.alpha, beta=args.beta,
                      gpu=args.gpu).to(device)
    else:
        raise ValueError('Incorrect Model Type! Choose one from the list: [vgg, cnn, linear]')
    model = model.double()
    model.multi_label = multi_label
    model.learning_rate = args.lr
    model.epoch = args.epoch
    model.batch_size = batch_size
    model.batch_size_2 = batch_size_unlabel
    if model.batch_size > train_dataset.data.shape[0]:
        model.batch_size = train_dataset.data.shape[0]
    if model.batch_size_2 > test_dataset.data.shape[0]:
        model.batch_size_2 = test_dataset.data.shape[0]
    print('Initial learning rate: {}'.format(args.lr))
    print('alpha = {}, beta = {}'.format(args.alpha, args.beta))
    start_time = time.time()
    train_model(model, train_dataset, args, test_dataset, multi_label, parallel_mode)
    print('time: {:.4f}'.format(time.time() - start_time))
    print('Finish training process!')
    evaluation(model, test_dataset)


def batch(iterable, data=None, n=1, n2=1):
    l = len(iterable)
    if data == None:
        iterable.shuffle()
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    else:
        iterable.shuffle()
        data.shuffle()
        l2 = len(data)
        if n < l:
            for ndx in range(0, min(int(l2 / n2), int(l / n))):
                yield (iterable[ndx * n:min(ndx * n + n, l)], data[ndx * n2:min(ndx * n2 + n2, l2)])
        else:
            for ndx in range(0, min(int(l2 / n2), int(l / n))):
                yield (iterable[:], data[ndx * n2:min(ndx * n2 + n2, l2)])


def train_model(model, train_loader, args, all_loader=None, multi_label=True, parallel_mode=False):
    epoch = 0
    cur_iter = 0
    momentum = 0.95
    clf_optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=momentum)
    clf_optimizer = LARC(clf_optimizer)
    stage = 1
    start_time = time.time()
    while epoch < model.epoch:
        epoch += 1
        predicted_label = []
        truth = []
        loss_mean = []
        l2_mean = []
        l3_mean = []
        prob = []
        if parallel_mode:
            parallel_model = torch.nn.DataParallel(model)  # Encapsulate the model
        else:
            parallel_model = model
        for train_sample, test_sample in batch(train_loader, all_loader,  model.batch_size, model.batch_size_2):
            clf_optimizer.zero_grad()
            x = torch.tensor(train_sample['data']).double().to(model.device)
            x2 = torch.tensor(test_sample['data']).double().to(model.device)
            labels = torch.tensor(train_sample['label']).double().to(model.device)
            x_v2 = torch.tensor(train_sample['data2']).double().to(model.device)
            x2_v2 = torch.tensor(test_sample['data2']).double().to(model.device)
            loss, prediction, _ = parallel_model(x, x2, labels=labels, stage=stage, x2_v2=x2_v2, x_v2=x_v2,
                                                 multi_label=multi_label)
            loss_mean.append(loss[0].mean())
            l2_mean.append(loss[1].mean())
            l3_mean.append(loss[2].mean())
            loss = (loss[0] + loss[1] + loss[2]).mean()
            clf_optimizer.zero_grad()
            loss.backward()
            clf_optimizer.step()
            cur_iter += 1
            if not multi_label:
                predicted = np.zeros_like(prediction.data.cpu())
                _, index = torch.max(prediction.data, 1)
                predicted[np.arange(index.shape[0]), index.cpu().numpy()] = 1
            else:
                predicted = np.array([[1 if a[j] > 0.4 else 0 for j in range(labels.shape[1])] for a in prediction.data])
            predicted_label = predicted_label + list(predicted)
            truth = truth + list(labels.cpu().numpy())
            prob = prob + list(prediction.data.cpu().numpy())
            torch.save(model.state_dict(), 'model.ckpt')
        if epoch % 10 == 0:
            prob = np.array(prob)
            f1 = f1_score(np.array(truth), np.array(predicted_label), average='weighted')
            auc = roc_auc_score(np.array(truth), prob)
            print('Epoch [{}/{}], time: {:.4f}, CLF Loss: {:.4f}, Lu: {:.4f},  Ls: {:.4f}, F1:{:.4f}, AUC: {:.4f}'.format(
                epoch, model.epoch, time.time() - start_time, torch.mean(torch.tensor(loss_mean)),
                torch.mean(torch.tensor(l2_mean)), torch.mean(torch.tensor(l3_mean)), f1, auc))
            start_time = time.time()
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_{}_{}_{}.ckpt'.format(args.data, epoch, args.hidden))


def evaluation(model, test_loader):
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
    model.batch_size = 50
    with torch.no_grad():
        total = 0
        predicted_label = []
        prob = []
        truth = []
        for sample_batched in batch(test_loader, n=model.batch_size):
            x = torch.tensor(sample_batched['data']).double().to(model.device)
            labels = torch.tensor(sample_batched['label']).double()
            x_v2 = torch.tensor(sample_batched['data2']).double().to(model.device)
            _, prediction, _ = model(x, x_v2=x_v2, stage=3, multi_label=model.multi_label)
            if not model.multi_label:
                predicted = np.zeros_like(prediction.data.cpu())
                _, index = torch.max(prediction.data, 1)
                predicted[np.arange(index.shape[0]).reshape(-1, 1), index.cpu().numpy().reshape(-1, 1)] = 1
            else:
                predicted = np.array([[1 if a[j] > 0.4 else 0 for j in range(labels.shape[1])] for a in prediction.data])
            predicted_label = predicted_label + list(predicted)
            truth = truth + list(labels.cpu().numpy())
            prob = prob + list(prediction.data.cpu().numpy())
            total += labels.size(0)
        length = test_loader.label.shape[0]
        f1 = f1_score(np.array(truth), np.array(predicted_label), average='weighted')
        auc = roc_auc_score(np.array(truth), np.array(prob))
        print('F1 scores of the model on the {} test documents: {:.4f}'.format(length, f1))
        print('AUC scores of the model on the {} test documents: {:.4f}'.format(length, auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HeroCon algorithm')
    parser.add_argument('-d', dest='data', type=str, default='scene', help='which dataset is used for demo')
    parser.add_argument('-g', dest='gpu', type=int, default=0, help='the index of the gpu to use')
    parser.add_argument('-lr', dest='lr', type=float, default=0.05, help='The initial learning rate')
    parser.add_argument('-e', dest='epoch', type=int, default=250, help='the total epoch for training')
    parser.add_argument('-hid', dest='hidden', type=int, default=512, help='the size of the hidden representation')
    parser.add_argument('-f', dest='fold', type=int, default=0, help='the index of 5 folds cross validation (0-4)')
    parser.add_argument('-m', dest='model', type=str, default='vgg', help='the index of 5 folds cross validation (0-4)')
    parser.add_argument('-alpha', dest='alpha', type=float, default=0.01, help='the value of alpha')
    parser.add_argument('-beta', dest='beta', type=float, default=0.01, help='the value of beta')
    args = parser.parse_args()
    print(args)
    for i in range(5):
        args.fold = i
        print('Round {}'.format(i+1))
        main(args)
        print('\n\n')
