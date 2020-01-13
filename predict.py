import argparse

from dataset import *
from models import FlowNetS, Unsupervised, LightFlowNet, PWC_Net
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="inference", type=str, metavar='DIR', help='path to get images')
    parser.add_argument("--unsup", help="use weights obtained of unsupervised training", action="store_true")
    parser.add_argument('--model', default='flownet', type=str, help='the model to be used either with or without '
                                                                     'supervision (flownet, lightflownet, pwc_net)')
    args = parser.parse_args()
    file_names = sorted(os.listdir(args.path))

    if args.unsup:
        mymodel = Unsupervised(conv_predictor=args.model)
        path = os.path.join("Unsupervised", type(mymodel.predictor).__name__)
    else:
        if "light" in args.model:
            mymodel = LightFlowNet()
        elif "pwc" in args.model:
            mymodel = PWC_Net()
        else:
            mymodel = FlowNetS()

        path = type(mymodel).__name__

    os.makedirs(os.path.join("result", path), exist_ok=True)
    mymodel.load_state_dict(torch.load(os.path.join("model_weight", path, 'best_weight.pt'), map_location=device)['model_state_dict'])
    if args.unsup:
        mymodel = mymodel.predictor
    mymodel.eval()
    frames_transforms = albu.Compose([
        albu.Normalize((0., 0., 0.), (1., 1., 1.)),
        ToTensor()
    ])
    for i in range(0, len(file_names) - 1, 2):
        frame1 = cv2.imread(os.path.join(args.path, file_names[i]))
        h, w = frame1.shape[:2]
        frame2 = cv2.imread(os.path.join(args.path, file_names[i + 1]))
        frame1 = frames_transforms(image=frame1)['image']
        frame2 = frames_transforms(image=frame2)['image']
        frames = torch.cat((frame1, frame2), dim=0)
        frames = torch.unsqueeze(frames, dim=0)

        with torch.no_grad():
            flow = mymodel(frames)[0]

        pred_flo = F.interpolate(flow, (h, w), mode='bilinear', align_corners=False)[0]
        pred_flo = computeImg(pred_flo.cpu().numpy(), verbose=True, savePath=os.path.join("result", path,
                                                                                         'predicted_flow' + str(
                                                                                             i//2 + 1) + '.png'))
