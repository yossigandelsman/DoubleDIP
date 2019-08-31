# Double Dip

Official implementation of the paper ["Double-DIP":
Unsupervised Image Decomposition via Coupled Deep-Image-Priors](http://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/resources/DoubleDIP.pdf).

Paper: http://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/resources/DoubleDIP.pdf


Project page: http://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/

----------
![sketch](/figs/Decomposition.png)
----------

If you find our work useful in your research or publication, please cite it:

```
@article{DoubleDIP,
author = {Gandelsman, Yossi and Shocher, Assaf and Irani, Michal},
year = {2019},
month = {6},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
title = {"Double-DIP": Unsupervised Image Decomposition via Coupled Deep-Image-Priors}
}
```
----------

## Further comments:
The airlight estimation in the dehazing part of the code uses the code provided by ["Blind Dehazing Using Internal Patch Recurrence"](https://github.com/YuvalBahat/Dehazing-Airlight-estimation).

The saliency detection that is used for segmentation hints provided by [Context-Aware Saliency Detection](https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/Saliency/Saliency.html), by Gofman et al. 
After applying this saliency detection, we thresholded it using `bg_fg_prep.py`.

The code is provided as-is for academic use only and without any guarantees. Please contact the author to report any bugs. 
