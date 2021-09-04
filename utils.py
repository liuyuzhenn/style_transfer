import argparse

def get_args():
    parse = argparse.ArgumentParser('Train GAN!!')
    parse.add_argument('-s', '--style_image', dest='simg', help='style image', type=str, required=True)
    parse.add_argument('-c', '--content_image', dest='cimg', help='content image', type=str, required=True)
    parse.add_argument('-o', '--out', default=r'./out.jpg', help='specify result\'s path', type=str)
    parse.add_argument('-sw', '--style_weight', dest='sw', default=5e5, help='style loss weigth', type=float)
    parse.add_argument('-cw', '--content_weight', dest='cw', default=1, help='content loss weight', type=float)
    parse.add_argument('-i', '--iteration', default=300, help='iterations', type=int)
    return parse.parse_args()