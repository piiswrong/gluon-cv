import matplotlib
matplotlib.use('Agg')

import argparse, time, logging, datetime

import numpy as np
import mxnet as mx
import gluoncv as gcv

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--gpus', type=str, default='',
                    help='Ordinal of gpus to use.')
parser.add_argument('--model', type=str, default='resnet',
                    help='model to use. options are resnet and wrn. default is resnet.')
parser.add_argument('--teacher', type=str, default='resnet',
                    help='teacher model to distill from')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='period in epoch for learning rate decays. default is 0 (has no effect).')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--drop-rate', type=float, default=0.0,
                    help='dropout rate for wide resnet. default is 0.')
parser.add_argument('--temperature', type=float, default=20,
                    help='Temperature of teacher softmax.')
parser.add_argument('--hard-weight', type=float, default=0.5,
                    help='Weight of hard loss.')
parser.add_argument('--mixup', type=float, default=0,
                    help='Alpha value for mixup training')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are imperative, hybrid')
parser.add_argument('--save-period', type=int, default=10,
                    help='period in epoch of model saving.')
parser.add_argument('--save-dir', type=str, default=None,
                    help='directory of saved models')
parser.add_argument('--resume-from', type=str,
                    help='resume training from the model')
opt = parser.parse_args()

# Logging
save_period = opt.save_period
save_dir = opt.save_dir
if not save_dir:
    save_dir = opt.model + '_' + '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
gcv.utils.makedirs(save_dir)

logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.FileHandler("%s/log.log"%save_dir))
logging.info(opt)

# Context
context = [mx.gpu(int(i)) for i in opt.gpus.strip().split(',')] if opt.gpus else [mx.cpu()]
num_gpus = len(context)
num_workers = opt.num_workers

# Model
classes = 10
model_name = opt.model
if model_name.startswith('cifar_wideresnet'):
    kwargs = {'classes': classes,
              'drop_rate': opt.drop_rate}
else:
    kwargs = {'classes': classes}
net = get_model(model_name, **kwargs)
if opt.resume_from:
    net.load_params(opt.resume_from, ctx = context)
net.initialize(mx.init.Xavier(), ctx=context)

# Teacher
teacher_name = opt.teacher
if teacher_name.startswith('cifar_wideresnet'):
    kwargs = {'classes': classes,
              'drop_rate': opt.drop_rate}
else:
    kwargs = {'classes': classes}
teacher = get_model(teacher_name, pretrained=True, **kwargs)
# teacher.load_parameters('0.9592-cifar-cifar_wideresnet40_8-131-best.params')
teacher.collect_params().reset_ctx(ctx=context)

# Optimizer
optimizer = 'nag'
batch_size = opt.batch_size * max(1, num_gpus)
lr_decay = opt.lr_decay
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]
temperature = opt.temperature
hard_weight = opt.hard_weight

# Data IO
transform_train = transforms.Compose([
    gcv_transforms.RandomCrop(32, pad=4),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)


# Loss
class DistillationLoss(gluon.HybridBlock):
    def __init__(self, temperature, hard_weight, sparse_label=True, **kwargs):
        super(DistillationLoss, self).__init__(**kwargs)
        self._temperature = temperature
        self._hard_weight = hard_weight
        with self.name_scope():
            self.soft_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
            self.hard_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label)

    def hybrid_forward(self, F, output, label, soft_target):
        if self._hard_weight == 0:
            return (self._temperature**2)*self.soft_loss(output/self._temperature, soft_target)
        elif self._hard_weight == 1:
            return self.hard_loss(output, label)
        else:
            return (1-self._hard_weight)*(self._temperature**2)*self.soft_loss(output/self._temperature, soft_target) + self._hard_weight*self.hard_loss(output, label)

loss_fn = DistillationLoss(temperature, hard_weight, sparse_label=(opt.mixup == 0))

def mixup_data(data, label, alpha):
    label = label.one_hot(classes)

    length = data.shape[0]
    idx = np.random.permutation(length)
    lam = mx.nd.array(np.random.beta(alpha, alpha, length))

    lam = lam.reshape((length,) + (1,) * (len(data.shape) - 1))
    data = lam * data + (1-lam) * data[idx]
    lam = lam.reshape((length,) + (1,) * (len(label.shape) - 1))
    label = lam * label + (1-lam) * label[idx]

    return data, label

def test(ctx, model, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [model(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
    metric = mx.metric.Accuracy()
    train_metric = mx.metric.Accuracy()
    hard_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    soft_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    train_history = TrainingHistory(['training-error', 'validation-error'])

    iteration = 0
    lr_decay_count = 0

    best_val_score = 0

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        metric.reset()
        train_loss = 0
        num_batch = len(train_data)
        alpha = 1

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, (data, label) in enumerate(train_data):
            if opt.mixup > 0:
                data, label = mixup_data(data, label, opt.mixup)
            data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)

            prob = [nd.softmax(teacher(X)/temperature) for X in data]

            with ag.record():
                output = [net(X) for X in data]
                loss = [loss_fn(yhat, l, p) for yhat, p, l in zip(output, prob, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in loss])

            train_metric.update(label, output)
            name, acc = train_metric.get()
            iteration += 1

        train_loss /= batch_size * num_batch
        name, acc = train_metric.get()
        name, val_acc = test(ctx, net, val_data)
        train_history.update([1-acc, 1-val_acc])
        train_history.plot(save_path='%s/%s_history.png'%(save_dir, model_name))
        logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
            (epoch, acc, val_acc, train_loss, time.time()-tic))

        if val_acc > best_val_score:
            best_val_score = val_acc
            net.save_parameters('%s/%.4f-cifar-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        if save_period and save_dir and (epoch + 1) % save_period == 0:
            net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epoch))

    if save_period and save_dir:
        net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
        teacher.hybridize(static_alloc=True, static_shape=True)
        loss_fn.hybridize(static_alloc=True, static_shape=True)

    logging.info('Teacher accuracy: %s: %f'%test(context, teacher, val_data))
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
