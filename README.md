# Trying to achieve PAG (perception-aligned gradients) with adversarial training

![PAG visualization](figs/PAG_visualization.png)
*Image source: 1805.12152. PAG visualization and comparison with gradients of a model trained regular way.*

Models with perception-aligned gradients are benifical in several obvious ways:
1. Interpretability. One can use gradients w.r.t. input data to produce model perception (or feature importance) heatmaps
2. Robustness. To achieve PAG one would usually perform adversarial training, which makes models much more robust

## Purpose of this repository
The main reason is developing a methodology of achieving PAG for one of my pet-projects.

PAG was introduced some years ago, though I haven't found any decent implementations and decided to implement it myself. I've chosen simple problem, dataset and model architecture so it'll be easier and quicker to perform all the experiments. 

## Dataset
I've chosen MNIST, which you can acquire at https://huggingface.co/datasets/ylecun/mnist

## Results

### Ordinarily trained model (no PAG)

|![](figs/ordinarily_trained_model/epoch100/0_7.png)|![](figs/ordinarily_trained_model/epoch100/1_2.png)|![](figs/ordinarily_trained_model/epoch100/2_1.png)|
|:---:|:---:|:---:|
|![](figs/ordinarily_trained_model/epoch100/3_0.png)|![](figs/ordinarily_trained_model/epoch100/4_4.png)|![](figs/ordinarily_trained_model/epoch100/7_9.png)|


### Adversarially trained model (PAG)

|![](figs/adversarially_trained_model/epoch100/0_7.png)|![](figs/adversarially_trained_model/epoch100/1_2.png)|![](figs/adversarially_trained_model/epoch100/2_1.png)|
|:---:|:---:|:---:|
|![](figs/adversarially_trained_model/epoch100/3_0.png)|![](figs/adversarially_trained_model/epoch100/4_4.png)|![](figs/adversarially_trained_model/epoch100/7_9.png)|

## References
1. Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).
2. Tsipras, Dimitris, et al. "Robustness may be at odds with accuracy." arXiv preprint arXiv:1805.12152 (2018).