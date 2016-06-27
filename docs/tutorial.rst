ZNN + AWS Tutorial
==================

The tutorial will help you learn how to use the ZNN AWS AMI by training a CNN to perform boundary detection on the ISBI 2012 dataset. In particular, the tutorial will focus on the training of the N4 network described in the paper `"Deep Neural Networks Segment Neuronal Membranes in Electron Microscopy Images" <https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images>`.

Since the python interface is more convenient to use, this tutorial only focuses on it.

1. Dataset Preparation
----------------------

Image format
````````````
ZNN accepts datasets that consist of 3D ``.tif`` or ``.h5`` image stacks.

============== ================= ===========
type            format            bit depth
============== ================= ===========
raw image       .tif              8
label image     .tif              32 or RGB
============== ================= ===========

* For training, you should prepare pairs of ``.tif`` files, one is a stack of raw images, the other is a stack of labeled images. A label is defined as a unique RGBA color.
* For forward pass, only the raw image stack is needed.

The ISBI 2012 challenge dataset is already provided in the AMI. It can be found in the following folder: ``/opt/znn-release/dataset/ISBI2012``. 

For further information about the ISBI 2012 dataset, please see the websites below:

* http://brainiac2.mit.edu/isbi_challenge/home
* http://journal.frontiersin.org/article/10.3389/fnana.2015.00142/full

Image configuration
```````````````````

ZNN requires a ``.spec`` file that provides the binding between the raw images and the labelled images (i.e. ground truth) in the dataset.

The image pairs are defined as a **Sample**.

The ``.spec`` file format allows you to specify multiple files as inputs (stack images) and outputs (ground truth labels) for a given experiment. A binding of inputs to outputs is called a sample.

The following code can be found in the ``dataset.spec`` file provided with the ISBI 2012 dataset (see folder ``/opt/znn-release/dataset/ISBI2012``):
::
    # samples example
    # the [image] sections indicate the network inputs
    # format should be gray images with any bit depth.
    #
    # input preprocessing types:
    # standard2D: minus by mean and than normalize by standard deviation
    # standard3D: normalize for the whole 3D volume
    # symetric_rescale: rescale to [ -1, 1 ]
    #
    # [image1]
    # fnames =  path/of/image1.tif/h5,
    #           path/of/image2.tif/h5
    # pp_types = standard2D, none
    # is_auto_crop = yes
    #
    # the [label] sections indicate ground truth of network outputs
    # format could be 24bit RGB or gray image with any bit depth.
    # the mask images should be binary image with any bit depth.
    # only the voxels with gray value greater than 0 is effective for training.
    #
    # [label1]
    # fnames = path/of/image3.tif/h5,
    #          path/of/image4.tif/h5
    # preprocessing type: one_class, binary_class, none, affinity
    # pp_types = binary_class, binary_class
    # fmasks = path/of/mask1.tif/h5,
    #	   path/of/mask2.tif/h5
    #
    # [sample] section indicates the group of the corresponding input and output labels
    # the name should be the same with the one in the network config file
    #
    # [sample1]
    # input1 = 1
    # input2 = 2
    # output1 = 1
    # output2 = 2
    
    [image1]
    fnames = ../dataset/ISBI2012/train-volume.tif
    pp_types = standard2D
    is_auto_crop = yes
    
    [label1]
    fnames = ../dataset/ISBI2012/train-labels.tif
    pp_types = auto
    is_auto_crop = yes
    fmasks =
    
    [sample1]
    input = 1
    output = 1
    
    [image2]
    fnames = ../dataset/ISBI2012/test-volume.tif
    pp_types = standard2D
    is_auto_crop = yes
    
    [sample2]
    input = 2

2. Network Architecture Configuration
-------------------------------------

We have a custom file format ``.znn`` for specifying the layout of your neural network. It works based on a few simple concepts. 

1. Each of the input nodes of the network represent an image stack.
2. The network consists of layers whose size can be individually specified. 
3. The edge betwen the layers specify not only the data transfer from one layer to another (e.g. one to one, or fully connected), they also prescribe a transformation, e.g. a filter or weight, to be applied. 
4. After all the weights or filters have been applied, the inputs are summed and a pixel-wise transfer function (e.g. a `sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function>`_ or `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_) is applied.
5. The type of the edges determines if the layers its connecting is a one-to-one mapping or is fully connected. For example, a convolution type will result in fully connected layers.
6. The output layer represents whatever you're training the network to do. One common output is the predicted labels for an image stack as a single node.

The following code is present in ``N4.znn``, which can be found in folder ``/opt/znn-release/networks``:
::
    nodes input
    type input
    size 1
    
    edges conv1
    type conv
    init xavier
    size 1,4,4
    stride 1,1,1
    input input
    output nconv1
    
    nodes nconv1
    type transfer
    function rectify_linear
    size 48
    
    edges pool1
    type max_filter
    size 1,2,2
    stride 1,2,2
    input nconv1
    output npool1
    
    nodes npool1
    type sum
    size 48
    
    edges conv2
    type conv
    init xavier
    size 1,5,5
    stride 1,1,1
    input npool1
    output nconv2
    
    nodes nconv2
    type transfer
    function rectify_linear
    size 48
    
    edges pool2
    type max_filter
    size 1,2,2
    stride 1,2,2
    input nconv2
    output npool2
    
    nodes npool2
    type sum
    size 48
    
    edges conv3
    type conv
    init xavier
    size 1,4,4
    stride 1,1,1
    input npool2
    output nconv3
    
    nodes nconv3
    type transfer
    function rectify_linear
    size 48
    
    edges pool3
    type max_filter
    size 1,2,2
    stride 1,2,2
    input nconv3
    output npool3
    
    nodes npool3
    type sum
    size 48
    
    edges conv4
    type conv
    init xavier
    size 1,4,4
    stride 1,1,1
    input npool3
    output nconv4
    
    nodes nconv4
    type transfer
    function rectify_linear
    size 48
    
    edges pool4
    type max_filter
    size 1,2,2
    stride 1,2,2
    input nconv4
    output npool4
    
    nodes npool4
    type sum
    size 48
    
    edges conv5
    type conv
    init xavier
    size 1,3,3
    stride 1,1,1
    input npool4
    output nconv5
    
    nodes nconv5
    type transfer
    function rectify_linear
    size 200
    
    edges conv6
    type conv
    init xavier
    size 1,1,1
    stride 1,1,1
    input nconv5
    output output
    
    nodes output
    type transfer
    function linear
    size 2

The ``.znn`` file is comprised of two primary objects -- nodes and edges. An object declaration consists of the type ``nodes`` or ``edges`` followed by its name on a new line followed by its parameters.

3. Training
-----------

Now that you've set up your training and validation datasets in your ``.spec`` file and have designed a neural network in your ``.znn`` file, 
it's time to tell the network exactly what to do. We do this via a ``.cfg`` configuration file.

Parameter configuration
```````````````````````
The training and forward parameters of the network can be set using a configuration file (`example <https://raw.githubusercontent.com/seung-lab/znn-release/abd05db3a97db1e39e437927746508357665bdde/python/config.cfg>`_). 

The configuration file uses the commonly used `Python ConfigParser <https://docs.python.org/2/library/configparser.html>`_. Consult that link for detailed information on acceptable syntax.
The ``.cfg`` file uses ``[sections]`` to ecapsulate different parameter sets. In the past, we used to use multiple sections, but now we just use one called ``[parameters]``.

Run a training
``````````````
After setting up the configuration file, you can now train your networks. 

Make sure you run the following command from within the `znn-release/python` directory. This is a limitation that can be fixed in future releases.
::
    python train.py -c path/of/config.cfg 

Resume a training
`````````````````
Since the network is periodically saved, we can resume training whenever we want to. By default, ZNN will automatically resume the latest training net (``net_current.h5``) in a folder, which was specified by the ``train_net`` parameter in the configuration file. 

To resume training a specific network, we can use the seeding function:
::
    python train.py -c path/of/config.cfg -s path/of/seed.h5

Transfer learning
`````````````````
Sometimes, we would like to utilize a trained network. If the network architectures of trained and initialized network are the same, we call it ``Loading``. Otherwise, we call it ``Seeding``, in which case the trained net is used as a seed to initialize part of the new network. Our implementation merges ``Loading`` and ``Seeding``. Just use the synonymous ``-s`` or ``--seed`` command line flags. 
::
    python train.py -c path/of/config.cfg -s path/of/seed.h5

Forward Pass
------------
run the following command:
::
    python forward.py -c path/of/config.cfg
if you are running forward pass intensively for a large image stack, it is recommanded to recompile python core using `DZNN_DONT_CACHE_FFTS`. Without caching FFTS, you can use a large output size, which reuse a lot of computation and speed up your forward pass.

NOTE: If your forward pass aborts without writing anything, try reducing the output size, as you may have run out of memory.
