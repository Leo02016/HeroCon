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

### More dataset
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* 
