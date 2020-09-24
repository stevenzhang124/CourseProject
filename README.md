# IERG6200 Course Project

## Directory tree
'''
+---data
+---dataset
|   +---amazon
|   |   \---images
|   +---dslr
|   |   \---images
|   \---webcam
|       \---images
+---model
|   \---__init__.py
    \---backbone.py
    \---mmd.py
    \---models.py
    \---Update.py
+---save
\---utils
    \---__init__.py
    \---Fed.py
    \---options.py      # configuration
    \---sampling.py     # data processing
+---main.py
'''

## TODO
+ Parameter fine-tuning
+ Explore methods to generate non-iid methods in sampling.py
+ Explore other methods except DDC
+ Extend to the case, where server also performs model training

## Note
+ Download the Office31 dataset from [here](https://pan.baidu.com/s/1o8igXT4#list/path=%2F). And then unrar it in ./dataset/.
+ This implementation does not simulate the communication between clients and server
+ The dataset Office31 itself is unbalanced