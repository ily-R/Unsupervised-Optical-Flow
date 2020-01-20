import argparse
import time
from dataset import *
from models import Unsupervised
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(seed=1)

PRINT_INTERVAL = 5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class AverageMeter(object):

    def __init__(self, keep_all=False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        if self.data is not None:
            self.data.append(value)
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def epoch(model, data, criterion, optimizer=None):
    model.eval() if optimizer is None else model.train()
    avg_loss = AverageMeter()
    avg_batch_time = AverageMeter()
    avg_smooth_loss = AverageMeter()
    avg_bce_loss = AverageMeter()

    tic = time.time()
    for i, (imgs, _) in enumerate(data):
        imgs = imgs.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred_flows, wraped_imgs = model(imgs)
            loss, bce_loss, smooth_loss = unsup_loss(pred_flows, wraped_imgs, imgs[:, :3, :, :])

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time = time.time() - tic
        tic = time.time()
        avg_bce_loss.update(bce_loss.item())
        avg_smooth_loss.update(smooth_loss.item())
        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)

        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'smooth_loss {smooth.val:5.1f} ({smooth.avg:5.1f})\t'
                  'bce_loss {bce.val:5.1f} ({bce.avg:5.1f})'.format(
                   "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                   smooth=avg_smooth_loss, bce=avg_bce_loss))

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg smooth_loss {smooth.avg:5.2f} \t'
          'Avg bce_loss {bce.avg:5.2f} \n'.format(
           batch_time=int(avg_batch_time.sum), loss=avg_loss,
           smooth=avg_smooth_loss, bce=avg_bce_loss))

    return avg_smooth_loss.avg, avg_bce_loss.avg, avg_loss.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default= '../FlyingChairs_release/data', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--model', default='flownet', type=str, help='the supervised model to be trained with ('
                                                                     'flownet, lightflownet, pwc_net)')
    parser.add_argument('--steps', default=600000, type=int, metavar='N', help='number of total steps to run')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', default=1.5e-5, type=float, metavar='LR', help='learning rate')
    parser.add_argument("--augment", help="perform data augmentation", action="store_true")
    parser.add_argument("--transfer", help="perform transfer learning from an already trained supervised model",
                        action="store_true")

    args = parser.parse_args()

    mymodel = Unsupervised(conv_predictor=args.model)
    path = os.path.join("Unsupervised", type(mymodel.predictor).__name__)
    loss_fnc = unsup_loss
    if args.transfer:
        best_model = torch.load(os.path.join("model_weight", type(mymodel.predictor).__name__, 'best_weight.pt'),
                                map_location=device)
        mymodel.predictor.load_state_dict(best_model['model_state_dict'])

    optim = torch.optim.Adam(mymodel.parameters(), args.lr)

    co_aug_transforms = None
    frames_aug_transforms = None

    frames_transforms = albu.Compose([
        albu.Normalize((0., 0., 0.), (1., 1., 1.)),
        ToTensor()
    ])

    if args.augment:
        if "FlyingChairs" in args.root:
            crop = albu.RandomSizedCrop((150, 384), 384, 512, w2h_ratio=512/384, p=0.5)
        elif "sintel" in args.root:
            crop = albu.RandomSizedCrop((200, 436), 436, 1024, w2h_ratio=1024/436, p=0.5)
        else:
            crop = albu.RandomSizedCrop((200, 400), 250, 250, w2h_ratio=1, p=0.5)
        co_aug_transforms = albu.Compose([
            crop,
            albu.Flip(),
            albu.ShiftScaleRotate()
        ])

        frames_aug_transforms = albu.Compose([
            albu.OneOf([albu.Blur(), albu.MedianBlur(), albu.MotionBlur()], p=0.5),

            albu.OneOf([albu.OneOf([albu.HueSaturationValue(), albu.RandomBrightnessContrast()], p=1),

                        albu.OneOf([albu.CLAHE(), albu.ToGray()], p=1)], p=0.5),
            albu.GaussNoise(),
        ])

    train, val, test = getDataloaders(args.batch_size, args.root, frames_transforms, frames_aug_transforms,
                                      co_aug_transforms)

    os.makedirs(os.path.join("Checkpoints", path), exist_ok=True)
    os.makedirs(os.path.join("model_weight", path), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs", path), flush_secs=20)

    tb_frames_train = next(iter(train))[0][0:1]
    tb_frames_val = next(iter(val))[0][0:1]
    tb_frames_test = next(iter(test))[0][0:1]

    starting_epoch = 0
    best_loss = 100000
    if os.path.exists(os.path.join("Checkpoints", path, 'training_state.pt')):
        checkpoint = torch.load(os.path.join("Checkpoints", path, 'training_state.pt'), map_location=device)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

    for e in range(starting_epoch, args.epochs):

        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")

        smooth_loss, bce_loss, total_loss = epoch(mymodel, train, loss_fnc, optim)

        torch.save({
            'epoch': e,
            'model_state_dict': mymodel.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optim.state_dict(),
        }, os.path.join("Checkpoints", path, 'training_state.pt'))

        smooth_loss_val, bce_loss_val, total_loss_val = epoch(mymodel, val, loss_fnc)

        if total_loss_val < best_loss:
            best_loss = total_loss_val
            torch.save({
                'model_state_dict': mymodel.state_dict(),
                'loss_val': total_loss_val, 'smooth_loss_val': smooth_loss_val, 'bce_loss_val': bce_loss_val,
                'loss': total_loss, 'smooth_loss': smooth_loss, 'bce_loss': bce_loss,
            }, os.path.join("model_weight", path, 'best_weight.pt'))

        smooth_loss_test, bce_loss_test, total_loss_test = epoch(mymodel, test, loss_fnc)
        with torch.no_grad():
            mymodel.eval()
            pred_flow = mymodel.predictor(tb_frames_train)[0]
            tb.add_image('train', disp_function(pred_flow, tb_frames_train[0]), e, dataformats='HWC')

            pred_flow = mymodel.predictor(tb_frames_val)[0]
            tb.add_image('val', disp_function(pred_flow, tb_frames_val[0]), e, dataformats='HWC')

            pred_flow = mymodel.predictor(tb_frames_test)[0]
            tb.add_image('test', disp_function(pred_flow, tb_frames_test[0]), e, dataformats='HWC')

        tb.add_scalars('loss', {"train": total_loss, "val": total_loss_val, "test": total_loss_test}, e)
        tb.add_scalars('smooth_loss', {"train": smooth_loss, "val": smooth_loss_val, "test": smooth_loss_test}, e)
        tb.add_scalars('bce_loss', {"train": bce_loss, "val": bce_loss_val, "test": bce_loss_test}, e)

        if "Flying" in args.root:
            if e % 100000 == 0:
                optim.param_groups[0]['lr'] *= 0.5
    tb.close()
