import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Weight_Net(nn.Module):
    def __init__(self, hid):
        super(Weight_Net, self).__init__()
        self.linear = nn.Linear(hid, hid)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)


class mlcpc(nn.Module):
    def __init__(self, num_class, d_feature=8520, d_2=112, gpu=0, alpha=0.1, beta=0.01):
        super(mlcpc, self).__init__()
        self.class_num = num_class
        self.hid = 200
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
        self.first_view = nn.Sequential(nn.Linear(d_feature, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
                                        nn.Linear(200, self.hid), nn.ReLU(inplace=True), nn.BatchNorm1d(self.hid))
        self.second_view = nn.Sequential(nn.Linear(d_2, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
                                         nn.Linear(200, self.hid), nn.ReLU(inplace=True), nn.BatchNorm1d(self.hid))
        self.e1 = nn.Sequential(nn.Linear(self.hid, self.hid), nn.ReLU(inplace=True), nn.BatchNorm1d(self.hid))
        self.e2 = nn.Sequential(nn.Linear(self.hid, self.hid), nn.ReLU(inplace=True), nn.BatchNorm1d(self.hid))
        self.final_layer = nn.Sequential(nn.Linear(self.hid * 2, self.class_num))
        self.cos = self.exp_cosine_sim
        self.weight_net = Weight_Net(self.hid)

    def forward(self, x, x2=None, x_v2=None, x2_v2=None, labels=None, stage=1, weighted_unsup=True, multi_label=True):
        if stage == 1:
            x1_v1_feature = self.first_view(x)
            x1_v2_feature =self.second_view(x_v2)
            x1_v1_e2 = self.e1(x1_v1_feature)
            x1_v2_e2 = self.e2(x1_v2_feature)
            all_feature = torch.cat([x1_v1_feature, x1_v2_feature], dim=1)
            prediction_prob = self.final_layer(all_feature)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(prediction_prob, labels)
            temp_labels = labels
            l2 = torch.tensor(0, device=self.device, dtype=torch.float64)
            l3 = torch.tensor(0, device=self.device, dtype=torch.float64)
            if self.alpha != 0:
                x2_v1_feature = self.first_view(x2)
                x2_v2_feature = self.second_view(x2_v2)
                x2_v1_e2 = self.e1(x2_v1_feature)
                x2_v2_e2 = self.e2(x2_v2_feature)
                v1_feature = torch.cat([x1_v1_e2, x2_v1_e2], dim=0)
                v2_feature = torch.cat([x1_v2_e2, x2_v2_e2], dim=0)
                if weighted_unsup:
                    sim = 1 / 2 * (torch.exp(1 - self.cosine_sim(self.weight_net(v1_feature), v2_feature)) +
                                   torch.exp(1 - self.cosine_sim(self.weight_net(v2_feature), v1_feature)))
                else:
                    sim = 1
                similarity = self.cos(v1_feature, v2_feature)
                size = all_feature.shape[0]
                l2 = torch.mean(torch.log((sim * similarity).sum(0)/(similarity.diag()*size*2)))
            if self.beta != 0:
                for i in range(self.class_num):
                    pos_idx = torch.where(temp_labels[:, i] == 1)[0]
                    if len(pos_idx) == 0:
                        continue
                    neg_idx = torch.where(temp_labels[:, i] != 1)[0]
                    pos_sample = all_feature[pos_idx, :]
                    neg_sample = all_feature[neg_idx, :]
                    size = neg_sample.shape[0] + 1
                    # distinguish whether it is multi-class problem or multi-label problem
                    if multi_label:
                        dist = self.hamming_distance_by_matrix(temp_labels)
                        pos_weight = 1 - dist[pos_idx, :][:, pos_idx] / self.class_num
                        neg_weight = dist[pos_idx, :][:, neg_idx]
                    else:
                        pos_weight = 1
                        neg_weight = 1
                    pos_dis = self.cos(pos_sample, pos_sample) * pos_weight
                    neg_dis = self.cos(pos_sample, neg_sample) * neg_weight
                    denominator = neg_dis.sum(1) + pos_dis
                    l3 += torch.mean(torch.log(denominator / (pos_dis * size)))
            if multi_label:
                return [loss, self.alpha * l2, self.beta * l3], torch.sigmoid(prediction_prob), None
            else:
                return [loss, self.alpha * l2, self.beta * l3], F.softmax(prediction_prob, dim=1), None
        else:
            x1_v1_feature = self.first_view(x)
            x1_v2_feature = self.second_view(x_v2)
            all_feature = torch.cat([x1_v1_feature, x1_v2_feature], dim=1)
            prediction_prob = self.final_layer(all_feature)
            if multi_label:
                return [None], torch.sigmoid(prediction_prob), None
            else:
                return [None], F.softmax(prediction_prob, dim=1), None

    def hamming_distance_by_matrix(self, labels):
        return torch.matmul(labels, (1 - labels).T) + torch.matmul(1 - labels, labels.T)

    def exp_cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.exp(torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature))

    def cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)


class vggcpc(nn.Module):
    def __init__(self, num_class, in_channels=3, gpu=0, alpha=0.1, beta=0.01):
        super(vggcpc, self).__init__()
        self.class_num = num_class
        self.hid = 128 * 7 * 7
        self.alpha = alpha
        self.beta = beta
        self.cfg = [32, 32, 'M', 32, 32, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M']
        layers = []
        self.batch_norm = True
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
        self.first_view = nn.Sequential(*layers)
        self.second_view = deepcopy(self.first_view)
        self.hid_layer_v1 = nn.Sequential(nn.Linear(self.hid, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512))
        self.hid_layer_v2 = nn.Sequential(nn.Linear(self.hid, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512))
        self.ev1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512))
        self.ev2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512))
        self.final_layer = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
                                         nn.Linear(512, self.class_num))
        self.exp_cos = self.exp_cosine_sim
        self.use_attention = True
        self.weight_net = Weight_Net(512)

    def forward(self, x, x2=None, x_v2=None, x2_v2=None, labels=None, stage=1, multi_label=True):
        if stage == 1:
            x1_v1_feature = self.first_view(x)
            x1_v2_feature = self.second_view(x_v2)
            x1_v1_feature = self.hid_layer_v1(x1_v1_feature.view(-1, self.hid))
            x1_v2_feature = self.hid_layer_v2(x1_v2_feature.view(-1, self.hid))
            x1_ve1 = self.ev1(x1_v1_feature)
            x1_ve2 = self.ev1(x1_v2_feature)
            all_feature = torch.cat([x1_v1_feature, x1_v2_feature], dim=1)
            prediction_prob = self.final_layer(all_feature)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(prediction_prob, labels)
            temp_labels = labels
            l2 = torch.tensor(0, device=self.device, dtype=torch.float64)
            l3 = torch.tensor(0, device=self.device, dtype=torch.float64)
            if self.alpha != 0:
                x2_v1_feature = self.first_view(x2)
                x2_v2_feature = self.second_view(x2_v2)
                x2_v1_feature = self.hid_layer_v1(x2_v1_feature.view(-1, self.hid))
                x2_v2_feature = self.hid_layer_v2(x2_v2_feature.view(-1, self.hid))
                x2_ve1 = self.ev1(x2_v1_feature)
                x2_ve2 = self.ev1(x2_v2_feature)
                v1_feature = torch.cat([x1_ve1, x2_ve1], dim=0)
                v2_feature = torch.cat([x1_ve2, x2_ve2], dim=0)
                similarity = self.exp_cos(v1_feature, v2_feature)
                sim = 1 / 2 * (torch.exp(1 - self.cosine_sim(self.weight_net(v1_feature), v2_feature)) +
                               torch.exp(1 - self.cosine_sim(self.weight_net(v2_feature), v1_feature)))
                # sim = 1
                size = all_feature.shape[0]
                l2 = torch.mean(torch.log((sim * similarity).sum(0)/(similarity.diag() * size)))
            if self.beta != 0:
                for i in range(self.class_num):
                    pos_idx = torch.where(temp_labels[:, i] == 1)[0]
                    if len(pos_idx) == 0:
                        continue
                    neg_idx = torch.where(temp_labels[:, i] != 1)[0]
                    if len(neg_idx) == 0:
                        continue
                    pos_sample = all_feature[pos_idx, :]
                    neg_sample = all_feature[neg_idx, :]
                    size = neg_sample.shape[0]
                    if multi_label:
                        dist = self.hamming_distance_by_matrix(temp_labels)
                        pos_weight = 1 - dist[pos_idx, :][:, pos_idx] / self.class_num
                        neg_weight = dist[pos_idx, :][:, neg_idx]
                    else:
                        pos_weight = 1
                        neg_weight = 1
                    pos_dis = self.exp_cos(pos_sample, pos_sample) * pos_weight
                    neg_dis = self.exp_cos(pos_sample, neg_sample) * neg_weight
                    denominator = neg_dis.sum(1) + pos_dis
                    l3 += torch.mean(torch.log(denominator / (pos_dis * size)))
                    if l3.isinf() or l3.isnan():
                        print('inf detected')
            if multi_label:
                return [loss, self.alpha * l2, self.beta * l3], torch.sigmoid(prediction_prob), None
            else:
                return [loss, self.alpha * l2, self.beta * l3], F.softmax(prediction_prob, dim=1), None
        else:
            x1_v1_feature = self.first_view(x)
            x1_v2_feature = self.second_view(x_v2)
            x1_v1_feature = self.hid_layer_v1(x1_v1_feature.view(-1, self.hid))
            x1_v2_feature = self.hid_layer_v2(x1_v2_feature.view(-1, self.hid))
            all_feature = torch.cat([x1_v1_feature, x1_v2_feature], dim=1)
            prediction_prob = self.final_layer(all_feature)
            if multi_label:
                return [None], torch.sigmoid(prediction_prob), None
            else:
                return [None], F.softmax(prediction_prob, dim=1), None

    def hamming_distance_by_matrix(self, labels):
        return torch.matmul(labels, (1-labels).T) + torch.matmul(1-labels, labels.T)

    def exp_cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.exp(torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature))

    def cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.matmul(x1, x2.t()) / ((w1 * w2.t()) * temperature)


class cnncpc(nn.Module):
    def __init__(self, num_class, d_feature=728, in_channel=1, gpu=0, alpha=0.1, beta=0.01):
        super(cnncpc, self).__init__()
        self.class_num = num_class
        self.hid = 32 * 13 * 13
        self.d_feature = d_feature
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
        self.first_view = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=4, stride=2))
        self.second_view = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=4, stride=2))
        self.hid_layer_v1 = nn.Sequential(nn.Linear(self.hid, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512))
        self.hid_layer_v2 = nn.Sequential(nn.Linear(self.hid, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512))
        self.final_layer = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
                                         nn.Linear(512, self.class_num))
        self.ev1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512))
        self.ev2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512))
        self.cos = self.exp_cosine_sim
        self.use_attention = True
        self.softmax = nn.Softmax(dim=1)
        self.weight_net = Weight_Net(512)

    def forward(self, x, x2=None, x_v2=None, x2_v2=None, labels=None, stage=1, weighted_unsup=True, multi_label=True):
        if stage == 1:
            x1_v1_feature = self.hid_layer_v1(self.first_view(x).view(-1, self.hid))
            x1_v2_feature = self.hid_layer_v2(self.second_view(x_v2).view(-1, self.hid))
            x1_ve1 = self.ev1(x1_v1_feature)
            x1_ve2 = self.ev1(x1_v2_feature)
            all_feature = torch.cat([x1_v1_feature, x1_v2_feature], dim=1)
            prediction_prob = self.final_layer(all_feature)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(prediction_prob, labels)
            temp_labels = labels
            l2 = torch.tensor(0, device=self.device, dtype=torch.float64)
            l3 = torch.tensor(0, device=self.device, dtype=torch.float64)
            if self.alpha != 0:
                x2_v1_feature = self.hid_layer_v1(self.first_view(x2).view(-1, self.hid))
                x2_v2_feature = self.hid_layer_v2(self.second_view(x2_v2).view(-1, self.hid))
                x2_ve1 = self.ev1(x2_v1_feature)
                x2_ve2 = self.ev1(x2_v2_feature)
                v1_feature = torch.cat([x1_ve1, x2_ve1], dim=0)
                v2_feature = torch.cat([x1_ve2, x2_ve2], dim=0)
                similarity = self.cos(v1_feature, v2_feature)
                if weighted_unsup:
                    sim = 1 / 2 * (torch.exp(1 - self.cosine_sim(self.weight_net(v1_feature), v2_feature)) +
                                   torch.exp(1 - self.cosine_sim(self.weight_net(v2_feature), v1_feature)))
                else:
                    sim = 1
                size = all_feature.shape[0]
                l2 = torch.mean(torch.log((sim * similarity).sum(0) / (similarity.diag() * size)))
            if self.beta != 0:
                for i in range(self.class_num):
                    pos_idx = torch.where(temp_labels[:, i] == 1)[0]
                    if len(pos_idx) == 0:
                        continue
                    neg_idx = torch.where(temp_labels[:, i] != 1)[0]
                    pos_sample = all_feature[pos_idx, :]
                    neg_sample = all_feature[neg_idx, :]
                    size = neg_sample.shape[0]
                    if multi_label:
                        dist = self.hamming_distance_by_matrix(temp_labels)
                        pos_weight = 1 - dist[pos_idx, :][:, pos_idx] / self.class_num
                        neg_weight = dist[pos_idx, :][:, neg_idx] / (temp_labels.sum() / temp_labels.shape[0])
                    else:
                        pos_weight = 1
                        neg_weight = 1
                    pos_dis = self.cos(pos_sample, pos_sample) * pos_weight
                    neg_dis = self.cos(pos_sample, neg_sample) * neg_weight
                    denominator = neg_dis.sum(1) + pos_dis
                    l3 += torch.mean(torch.log(denominator / (pos_dis * size)))
            if multi_label:
                return [loss, self.alpha * l2, self.beta * l3], torch.sigmoid(prediction_prob), None
            else:
                return [loss, self.alpha * l2, self.beta * l3], self.softmax(prediction_prob), None
        else:
            x1_v1_feature = self.first_view(x)
            x1_v1_feature = self.hid_layer_v1(x1_v1_feature.view(-1, self.hid))
            x1_v2_feature = self.second_view(x_v2)
            x1_v2_feature = self.hid_layer_v2(x1_v2_feature.view(-1, self.hid))
            all_feature = torch.cat([x1_v1_feature, x1_v2_feature], dim=1)
            prediction_prob = self.final_layer(all_feature)
            if multi_label:
                return [None], torch.sigmoid(prediction_prob), None
            else:
                return [None], self.softmax(prediction_prob), None

    def hamming_distance_by_matrix(self, labels):
        return torch.matmul(labels, (1-labels).T) + torch.matmul(1-labels, labels.T)

    def exp_cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.exp(torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature))

    def cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)
