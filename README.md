# Style Transfer
- Reference: [Image Style Transfer Using Convolutional Neural Networks
](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

### Usage
**generate image**
```sh
git clone --depth=1 git@github.com:liuyuzhenn/style_transfer.git
cd style_transfer
python .\transfer.py -s .\pic\style.jpg -c .\pic\content.jpg -o .\pic\out.jpg
```

**for details**
```sh
python .\transfer.py -h
```

### Demo
<div align=center>
<img src="https://github.com/liuyuzhenn/style_transfer/blob/master/pic/content.jpg" >
</div>

<div align=center>
<img src="https://github.com/liuyuzhenn/style_transfer/blob/master/pic/style.jpg">
</div>

<div align=center>
<img src="https://github.com/liuyuzhenn/style_transfer/blob/master/pic/out.jpg">
</div>