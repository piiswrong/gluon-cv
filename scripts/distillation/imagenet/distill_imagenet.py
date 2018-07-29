import argparse, time, logging, os, datetime

import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                    help='the training data')
parser.add_argument('--rec-train-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/train.idx',
                    help='the index of training data')
parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                    help='the validation data')
parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                    help='the index of validation data')
parser.add_argument('--use-rec', action='store_true',
                    help='use image record iter for data input. default is false.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
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
parser.add_argument('--lr-mode', type=str, default='step',
                    help='learning rate scheduler mode. options are step, poly and cosine.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='interval for periodic learning rate decays. default is 0 to disable.')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--last-gamma', action='store_true',
                    help='whether to initialize the gamma of the last BN layer in each bottleneck to zero')
parser.add_argument('--temperature', type=float, default=20,
                    help='Temperature of teacher softmax.')
parser.add_argument('--hard-weight', type=float, default=0.5,
                    help='Weight of hard loss.')
parser.add_argument('--mixup', type=float, default=0,
                    help='Alpha value for mixup training')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--teacher', type=str, default='resnet',
help='teacher model to distill from')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--save-frequency', type=int, default=10,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default=None,
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
opt = parser.parse_args()

# Logging
save_frequency = opt.save_frequency
save_dir = opt.save_dir
if not save_dir:
    save_dir = opt.model + '_' + '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
gcv.utils.makedirs(save_dir)

logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.FileHandler("%s/log.log"%save_dir))
logging.info(opt)


# Parameters
batch_size = opt.batch_size
classes = 1000
num_training_samples = 1281167

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers


# Optimization
lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
num_batches = num_training_samples // batch_size
lr_scheduler = LRScheduler(mode=opt.lr_mode, baselr=opt.lr,
                            niters=num_batches, nepochs=opt.num_epochs,
                            step=lr_decay_epoch, step_factor=opt.lr_decay, power=2,
                            warmup_epochs=opt.warmup_epochs)

optimizer = 'nag'
optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True


# Teacher
teacher_name = opt.teacher

kwargs = {'ctx': context, 'pretrained': True, 'classes': classes}
if teacher_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif teacher_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

if opt.last_gamma:
    kwargs['last_gamma'] = True

teacher = get_model(teacher_name, **kwargs)
teacher.cast(opt.dtype)


# Model
model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

if opt.last_gamma:
    kwargs['last_gamma'] = True

net = get_model(model_name, **kwargs)
net.cast(opt.dtype)


# Two functions for reading data from record file or raw images
def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, 224, 224),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = 256,
        data_shape          = (3, 224, 224),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return train_data, val_data, batch_fn

def get_data_loader(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256, keep_ratio=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, val_data, batch_fn

if opt.use_rec:
    train_data, val_data, batch_fn = get_data_rec(opt.rec_train, opt.rec_train_idx,
                                                  opt.rec_val, opt.rec_val_idx,
                                                  batch_size, num_workers)
else:
    train_data, val_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers)


# Loss


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

loss_fn = DistillationLoss(opt.temperature, opt.hard_weight, sparse_label=(opt.mixup == 0))

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

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

def test(net, ctx, val_data):
    if opt.use_rec:
        val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

def train(ctx):
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    best_val_score = 1

    for epoch in range(opt.num_epochs):
        tic = time.time()
        if opt.use_rec:
            train_data.reset()
        acc_top1.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)

            prob = [nd.softmax(teacher(X)/opt.temperature) for X in data]

            with ag.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                loss = [loss_fn(yhat, l, p) for yhat, p, l in zip(outputs, prob, label)]
            for l in loss:
                l.backward()
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)

            acc_top1.update(label, outputs)
            if opt.log_interval and not (i+1)%opt.log_interval:
                _, top1 = acc_top1.get()
                err_top1 = 1-top1
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\ttop1-err=%f\tlr=%f'%(
                             epoch, i, batch_size*opt.log_interval/(time.time()-btic), err_top1,
                             trainer.learning_rate))
                btic = time.time()

        _, top1 = acc_top1.get()
        err_top1 = 1-top1
        throughput = int(batch_size * i /(time.time() - tic))

        err_top1_val, err_top5_val = test(net, ctx, val_data)

        logging.info('[Epoch %d] training: err-top1=%f'%(epoch, err_top1))
        logging.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput, time.time()-tic))
        logging.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

        if err_top1_val < best_val_score and epoch > 50:
            best_val_score = err_top1_val
            net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, opt.num_epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
        teacher.hybridize(static_alloc=True, static_shape=True)
        loss_fn.hybridize(static_alloc=True, static_shape=True)
    err_top1_val, err_top5_val = test(teacher, context, val_data)
    logging.info('Teacher: err-top1=%f err-top5=%f'%(err_top1_val, err_top5_val))
    train(context)

if __name__ == '__main__':
    main()
