import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
import torch
import torchvision as tv
from PIL import Image
from dataset_loaders import IMAGENET_MEAN, IMAGENET_STD, DAVIS17V2, LabelToLongTensor
import evaluation
import models
from local_config import config
from optparse import OptionParser


parser = OptionParser()
parser.add_option("--save", action="store_true", dest="save", default=None)
parser.add_option("--fps", action="store_true", dest="fps", default=None)
(options, args) = parser.parse_args()


torch.cuda.set_device(0)


def test_model(model, save):
    nframes = 128

    def image_read(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [tv.transforms.ToTensor(),
             tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        return transform(pic)

    def label_read(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [LabelToLongTensor()])
        label = transform(pic)
        return label

    datasets = {
        'DAVIS16_val': DAVIS17V2(config['davis_path'], '2016', 'val', image_read, label_read, nframes),
        'DAVIS17_val': DAVIS17V2(config['davis_path'], '2017', 'val', image_read, label_read, nframes)}

    for key, dataset in datasets.items():
        if key == 'DAVIS16_val':
            evaluator = evaluation.VOSEvaluator(dataset, 'cuda', save)
            print("\n-- DAVIS16 dataset initialization started.")
        elif key == 'DAVIS17_val':
            evaluator = evaluation.VOSEvaluator(dataset, 'cuda', save)
            print("\n-- DAVIS17 dataset initialization started.")
        result_fpath = os.path.join(config['output_path'])
        evaluator.evaluate(model, os.path.join(result_fpath, key))


def main():
    model = models.PMVOS(
        backbone=('resnet50s16', (True, ('layer4',), ('layer4',), ('layer2',), ('layer1',))))
    print("Network model {} loaded".format(model.__class__.__name__))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(torch.load('PMVOS_best.tar'))

    if options.save is True:
        test_model(model, save=True)
    if options.fps is True:
        test_model(model, save=False)


if __name__ == '__main__':
    main()
