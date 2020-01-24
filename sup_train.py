import argparse
import time
from dataset import *
from models import FlowNetS, PWC_Net, LightFlowNet
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(seed=1)

PRINT_INTERVAL = 50
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
    avg_EPE = AverageMeter()
    avg_AAE = AverageMeter()

    tic = time.time()
    for i, (imgs, flow) in enumerate(data):
        imgs = imgs.to(device)
        flow = flow.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(imgs)
            loss = criterion(outputs, flow)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ep_error, aa_error = evaluate(outputs[0].data, flow.data)
        avg_EPE.update(ep_error.item())
        avg_AAE.update(aa_error.item())
        batch_time = time.time() - tic
        tic = time.time()

        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)

        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'EPE {epe.val:5.4f} ({epe.avg:5.4f})\t'
                  'AAE {aae.val:5.4f} ({aae.avg:5.4f})'.format(
                   "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                   epe=avg_EPE, aae=avg_AAE))

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg EPE {epe.avg:5.4f} \t'
          'Avg AAE {aae.avg:5.4f} \n'.format(
           batch_time=int(avg_batch_time.sum), loss=avg_loss,
           epe=avg_EPE, aae=avg_AAE))

    return avg_EPE.avg, avg_AAE.avg, avg_loss.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../ChairsSDHom/data', type=str, metavar='DIR', help='path to '
                                                                                                        'dataset')
    parser.add_argument('--model', default='flownet', type=str, help='the model to be trained (flownet, '
                                                                     'lightflownet, pwc_net)')
    parser.add_argument('--steps', default=1700000, type=int, metavar='N', help='number of total steps to run')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', default=1e-5, type=float, metavar='LR', help='learning rate')
    parser.add_argument("--augment", help="perform data augmentation", action="store_true")

    args = parser.parse_args()

    if "light" in args.model:
        mymodel = LightFlowNet()
        loss_fnc = None
    elif "pwc" in args.model:
        mymodel = PWC_Net()
        loss_fnc = None
    else:
        mymodel = FlowNetS()
        loss_fnc = EPE_all

    path = type(mymodel).__name__
    mymodel.to(device)
    optim = torch.optim.Adam(mymodel.parameters(), args.lr)

    co_aug_transforms = None
    frames_aug_transforms = None

    frames_transforms = albu.Compose([
        albu.Normalize((0., 0., 0.), (1., 1., 1.)),
        ToTensor()
    ])

    if args.augment:
        if "Chairs" in args.root:
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
    train_length = len(train)
    epochs = args.steps // train_length
    tb_frames_train, tb_flow_train = next(iter(train))
    tb_frames_val, tb_flow_val = next(iter(val))
    tb_frames_test, tb_flow_test = next(iter(test))
    tb_frames_train, tb_flow_train = tb_frames_train[0:1].to(device), tb_flow_train[0]
    tb_frames_val, tb_flow_val = tb_frames_val[0:1].to(device), tb_flow_val[0]
    tb_frames_test, tb_flow_test = tb_frames_test[0:1].to(device), tb_flow_test[0]

    os.makedirs(os.path.join("Checkpoints", path), exist_ok=True)
    os.makedirs(os.path.join("model_weight", path), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs", path), flush_secs=20)

    starting_epoch = 0
    best_loss = 100000
    if os.path.exists(os.path.join("Checkpoints", path, 'training_state.pt')):
        checkpoint = torch.load(os.path.join("Checkpoints", path, 'training_state.pt'), map_location=device)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

    mile_stone1 = 1400000 // train_length
    mile_stone2 = 100000 // train_length
    for e in range(starting_epoch, epochs):
        
        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")
        print("learning rate : ", optim.param_groups[0]["lr"])
        avg_epe, avg_aae, loss = epoch(mymodel, train, loss_fnc, optim)

        torch.save({
            'epoch': e,
            'model_state_dict': mymodel.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optim.state_dict(),
        }, os.path.join("Checkpoints", path, 'training_state.pt'))

        avg_epe_val, avg_aae_val, loss_val = epoch(mymodel, val, loss_fnc)

        if loss_val < best_loss:
            print("---------saving new weights!----------")
            best_loss = loss_val
            torch.save({
                'model_state_dict': mymodel.state_dict(),
                'loss_val': loss_val, 'epe_val': avg_epe_val, 'aae_val': avg_aae_val,
                'loss': loss, 'epe': avg_epe, 'aae': avg_aae,
            }, os.path.join("model_weight", path, 'best_weight.pt'))

        avg_epe_test, avg_aae_test, loss_test = epoch(mymodel, test, loss_fnc)
        with torch.no_grad():
            mymodel.eval()

            pred_flow = mymodel(tb_frames_train)[0]
            tb.add_images('train', disp_function(pred_flow.data, tb_flow_train.data), e, dataformats='NHWC')

            pred_flow = mymodel(tb_frames_val)[0]
            tb.add_images('val', disp_function(pred_flow.data, tb_flow_val.data), e, dataformats='NHWC')

            pred_flow = mymodel(tb_frames_test)[0]
            tb.add_images('test', disp_function(pred_flow.data, tb_flow_test.data), e, dataformats='NHWC')

        tb.add_scalars('loss', {"train": loss, "val": loss_val, "test": loss_test}, e)
        tb.add_scalars('EPE', {"train": avg_epe, "val": avg_epe_val, "test": avg_epe_test}, e)
        tb.add_scalars('AAE', {"train": avg_aae, "val": avg_aae_val, "test": avg_aae_test}, e)

        if "Chairs" in args.root:
            if e >= mile_stone1 and optim.param_groups[0]['lr'] == 1e-5:
                optim.param_groups[0]['lr'] *= 0.5
            elif e % mile_stone2 == 0 and optim.param_groups[0]['lr'] != 1e-5:
                optim.param_groups[0]['lr'] *= 0.5

    tb.close()
