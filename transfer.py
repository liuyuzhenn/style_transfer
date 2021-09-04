import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
from loss import *
from utils import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
loader = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

def load_image(path):
    img = Image.open(path)
    w, h = img.size
    img = loader(img).unsqueeze(0)
    return img, w, h



class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean.view(-1, 1, 1).to(device)
        self.std = std.view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# desired depth layers to compute style/content losses :content_layers_default = ['conv_4']style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                            style_img, content_img,
                            content_layers=content_layers_default,
                            style_layers=style_layers_default):
    # normalization module
    normalization = Normalize(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = torch.optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == "__main__":
    args = get_args()
    style_path = args.simg
    content_path = args.cimg
    result_path = args.out
    style_weight = args.sw
    content_weight = args.cw
    iterations = args.iteration

    style_img,_,_ = load_image(style_path)
    content_img, w, h = load_image(content_path)
    style_img = style_img.to(device)
    content_img = content_img.to(device)

    cnn = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    img_begin = content_img.clone()

    img_out = run_style_transfer(cnn, cnn_normalization_mean, 
    cnn_normalization_std, content_img, style_img, img_begin, 
    iterations, style_weight, content_weight).detach().cpu().reshape((3, IMG_SIZE, IMG_SIZE))
    img_out = (img_out.numpy() * 255).astype(np.uint8)
    img_out = np.transpose(img_out, (1, 2, 0))[:,:,::-1]
    img_out = cv2.resize(img_out, (w, h))

    cv2.imwrite(result_path, img_out)