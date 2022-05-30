### Demo Code for HeroCon algorithm

### Dependencies
* PyTorch
* pandas


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
python main.py -d celeba -m linear -alpha 0.05 -beta 0.1 -e 300
```

### More datasets
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [XRMB](https://home.ttic.edu/~klivescu/XRMB_data/full/README)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
