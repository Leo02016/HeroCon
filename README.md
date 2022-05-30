### Demo Code for paper "Contrastive Learning with Complex heterogeneity" accepted by KDD 2022.

### Dependencies
* PyTorch
* Pandas
* CV2
* Sklearn


### Command
Multi-view Multi-label setting on Scene dataset 
```
python main.py -d scene -m linear -alpha 0.7 -beta 0.02 -e 200 -hid 100
```

Multi-view Multi-class setting on XRMB dataset
```
python main.py -d xrmb -m linear -alpha 5 -beta 0.01 -e 300 -hid 100
```

Multi-view Multi-class setting on CelebA dataset
```
python main.py -d mnist -m linear -alpha 0.1 -beta 1 -e 300 -hid 200
```

Multi-view Multi-label setting on CelebA dataset
```
python main.py -d celeba -m vgg -alpha 0.05 -beta 0.1 -e 300
```

### More datasets
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [XRMB](https://home.ttic.edu/~klivescu/XRMB_data/full/README)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


Please 
