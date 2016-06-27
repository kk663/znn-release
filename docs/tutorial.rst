ZNN + AWS Tutorial
==================

The tutorial will help you learn how to use the ZNN AWS AMI by training a CNN to perform boundary detection on the ISBI 2012 dataset. In particular, the tutorial will focus on the training of the N4 network described in the paper `"Deep Neural Networks Segment Neuronal Membranes in Electron Microscopy Images" <https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images>`_.

The tutorial focuses on usage of the python interface since it is more convenient to use.

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

The following code is present in ``N4.znn`` which can be found in folder ``/opt/znn-release/networks``:
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
The training and forward parameters of the network can be set using a configuration file. 

The configuration file uses the commonly used `Python ConfigParser <https://docs.python.org/2/library/configparser.html>`_. Consult that link for detailed information on acceptable syntax.
The ``.cfg`` file uses ``[sections]`` to ecapsulate different parameter sets. In the past, we used to use multiple sections, but now we just use one called ``[parameters]``.

The following code is present in ``config.cfg`` which can be found in folder ``/opt/znn-release/python``:
::
    [parameters]
    # general
    # specification file of network architecture
    fnet_spec = ../networks/N4.znn
    # file of data spec
    fdata_spec = ../dataset/test/dataset.spec
    # number of threads. if <=0, the thread number will be equal to
    # the number of concurrent threads supported by the implementation.
    num_threads = 0
    # data type of arrays: float32 or float64
    dtype = float32
    # type of network output: boundary or affinity
    out_type = boundary
    # Whether to record config and log files
    logging = no
    
    # train
    # saved network file name. will automatically add iteration number
    # saved file name example: net_21000.h5, net_current.h5
    # the net_current.h5 will always be the latest network
    train_net_prefix = ../experiments/piriform/N4/net
    # sample ID range for train
    # example: 2-3,7
    train_range = 2
    # sample ID range for validate/test during training
    # example: 1,4-6,8
    test_range = 1
    # dense output size of one forward pass: z,y,x
    # large output size can reduce the computational redundency
    # this parameter affects the memory consumption a lot.
    # keep an eye to the memory, if it occupies too much memory, reduce this outsz
    train_outsz = 1,100,100
    
    # mode: fft, direct, optimize
    # if optimize, znn will choose direct convolution or fft for each layer.
    # optimize will get the best performance, but it takes a few minutes at the beginning.
    # it is suggested to use fft for fast testing and forward pass, and use optimize for long-time training
    train_conv_mode = fft
    
    # cost function: square_loss, binomial_cross_entropy, softmax_loss, auto
    # auto mode will match the out_type: boundary-softmax_loss, affinity-binomial_cross_entropy
    cost_fn = auto
    # use malis weighting of gradient
    # Maximin affinity learning of image segmentation
    # http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation
    # For normal training, you don't need this.
    is_malis = no
    # type of malis normalization:
    # none: no normalization,
    # frac: segment fractional normalization
    # num : normalized by N (number of nonboundary voxels)
    # pair: normalized by N*(N-1)
    malis_norm_type = none
    
    # learning rate
    eta = 0.01
    # annealing factor
    anneal_factor = 0.997
    # number of iteration per learning rate annealing
    Num_iter_per_annealing = 100
    # momentum
    momentum = 0.9
    # weight decay
    weight_decay = 0
    
    # randomly transform patches to enrich training data, including rotation, fliping
    is_data_aug = yes
    # mirror the image region close to boundaries to get a full size output
    is_bd_mirror = yes
    # balance the boundary and non-boundary voxel
    # global: compute the weight in the whole image stack
    # patch: compute the balance weight for each patch
    rebalance_mode = global
    
    # standard IO format in Seunglab: https://docs.google.com/spreadsheets/d/1Frn-VH4VatqpwV96BTWSrtMQV0-9ej9soy6HXHgxWtc/edit?usp=sharing
    # if yes, will save the learning curve and network in one file
    # if no, will save them separatly. This will be backward compatable.
    # For new training, it is recommanded to use stdio
    is_stdio = yes
    # debug mode: yes, no
    # if yes, will output some internal information and save patches in network file.
    is_debug = no
    # check the patches, used in Travis-ci for automatic test
    is_check = no
    
    # number of iteration per output
    Num_iter_per_show = 100
    # number of iteration per validation/test during training
    Num_iter_per_test = 200
    # number of patches to run forward pass for validation/test
    # the larger the smoother of learning curve, but the slower the training
    test_num = 10
    # number of iteration per save
    Num_iter_per_save = 1000
    # maximum iteration
    Max_iter = 200000
    
    # forward
    # sample ID for forward pass, example: 2-3,8
    forward_range = 1
    # forward network
    forward_net = ../experiments/piriform/N4/net_current.h5
    # forward convolution mode: fft, direct, optimize
    # since optimization takes a long time, normally just use fft
    forward_conv_mode = fft
    # output size of one forward pass: z,y,x
    # the larger the faster, limited by the memory capacity.
    forward_outsz = 5,100,100
    # output file name prefix
    output_prefix = ../experiments/piriform/N4/out
    
Training the N4 network
```````````````````````
After setting up the configuration file, you can now train your network. You need to run training as root. Please enter ``sudo su`` in the terminal after you have ssh-ed to your AWS instance (the instance launched using the ZNN AWS AMI image). 

Make sure you run the following command from within the `/opt/znn-release/python` directory. This is a limitation that can be fixed in future releases.
::
    python train.py -c path/of/config.cfg 

Resume training the N4 network
``````````````````````````````
Since the network is periodically saved, we can resume training whenever we want to. By default, ZNN will automatically resume the latest training net (``net_current.h5``) in a folder, which was specified by the ``train_net`` parameter in the configuration file. 

To resume training a specific network, we can use the seeding function:
::
    python train.py -c path/of/config.cfg -s path/of/seed.h5

Transfer learning using the N4 network
``````````````````````````````````````
Sometimes, we would like to utilize a trained network. If the network architectures of trained and initialized network are the same, we call it ``Loading``. Otherwise, we call it ``Seeding``, in which case the trained net is used as a seed to initialize part of the new network. Our implementation merges ``Loading`` and ``Seeding``. Just use the synonymous ``-s`` or ``--seed`` command line flags. 
::
    python train.py -c path/of/config.cfg -s path/of/seed.h5

Forward Pass using the N4 network
---------------------------------
run the following command:
::
    python forward.py -c path/of/config.cfg
if you are running forward pass intensively for a large image stack, it is recommanded to recompile python core using `DZNN_DONT_CACHE_FFTS`. Without caching FFTS, you can use a large output size, which reuse a lot of computation and speed up your forward pass.

NOTE: If your forward pass aborts without writing anything, try reducing the output size, as you may have run out of memory.

4. Specification File Format
----------------------------

The ``.spec`` file format allows you to specify multiple files as inputs (stack images) and outputs (ground truth labels) for a given experiment. A binding of inputs to outputs is called a sample.

The file structure looks like this, where "N" in imageN can be any positive integer. Items contained inside angle brackets are <option1, option2> etc.
::
    [imageN]
    fnames = path/of/image1
             path/of/image2
    pp_types = <standard2D, none> # preprocess the image by subtracting mean
    is_auto_crop = <yes, no> # crop images to mutually fit and fit ground truth labels

    [labelN]
    fnames = path/of/label1
    pp_types = <one_class, binary_class, affinity, none>
    is_auto_crop = <yes, no>
    fmasks =  path/of/mask1
              path/of/mask1

    [sampleN]
    input = 1
    output = 1  

[imageN] options
````````````````
Declaration of source images to train on.

Required:

1. ``fnames``: Paths to image stack files.

Optional:

1. ``pp_types`` (preprocessing types): none (default), standard2D, standard3D, symetric_rescale  
::
    standard2D modifies the image by subtracting the mean and dividing by the standard deviation of the pixel values.
    standard3D normalize for whole 3D volume like standard2D  
    symmetric_rescale rescales to [ -1, 1 ]  

2. ``is_auto_crop``: no (default), yes 
    If the corresponding ground truth stack's images are not the same dimension as the image set (e.g. image A is 1000px x 1000px and label A is 100px x 100px), then the smaller image will be centered in the larger image and the larger image will be cropped around it.


[labelN] options
````````````````
Declaration of ground truth labels to evaluate training on.

Required:

1. ``fnames``: Paths to label stack files.

Optional:

1. ``pp_types`` (preprocessing types): none (default), one_class, binary_class, affinity

==================== =========================================================
 Preprocessing Type  Function
==================== =========================================================
 none                Don't do anything.
 one_class           Normalize values, threshold at 0.5
 binary_class        one_class + generate extra inverted version
 affinity            Generate X, Y, & Z stacks for training on different axes   
==================== =========================================================

2. ``is_auto_crop``: no (default), yes 
    If the corresponding ground truth stack's images are not the same dimension as the image set (e.g. image A is 1000px x 1000px and label A is 100px x 100px), then the smaller image will be centered in the larger image and the larger image will be cropped around it.

3. ``fmasks``: Paths to mask files
    fmasks are used like cosmetics to coverup damaged parts of images so that your neural net
    doesn't learn useless information. Pixel values greater than zero are on. That is to say, white is on, black is off. The same file types are supported as for regular images.

[sampleN] options
`````````````````

Declaration of binding between images and labels. You'll use the sample number in your training configuration to decide which image sets to train on.

Required:

1. ``input``: (int > 0) should correspond to the N in an [imageN]. e.g. ``input: 1`` 
2. ``output``: (int > 0) should correspond to the N in a [labelN]. e.g. ``output: 1``

5. ZNN File Format
------------------

``nodes`` type declaration
``````````````````````````

Note: In the Description column for functions, the relevant funciton_args are presented as:
``[ comma,seperated,variables | default,values,here ]``

================ =========== =================== ================================================================
 Property         Required    Options             Description                                                    
================ =========== =================== ================================================================
 nodes            Y           $NAME               Symbolic identifier for other layers to reference. The names "input" and "output" are special and represent the input and output layers of the entire network.
 type             Y           sum                 Perform a simple weighted summing of the inputs to this node.
 ..               ..          transfer            Perform a summation of the input nodes and then apply a transfer function (c.f. function).
 function         N           linear              Line. ``[ slope,intercept | 1,1 ]``
 ..               ..          rectify_linear      Rectified Linear Unit (ReLU)
 ..               ..          tanh                Hyperbolic Tangent. ``[ amplitude,frequency | 1,1 ]``
 ..               ..          soft_sign           x / (1 + abs(x))
 ..               ..          logistics           Logistic function aka sigmoid. Has gradient.
 ..               ..          forward_logistics   Same as "logistics" but without a gradient?
 function_args    N           $VALUES             Input comma seperated values of the type appropriate for the selected function.
 size             Y           $POSTIVE_INTEGER    The number of nodes in this layer.
================ =========== =================== ================================================================

``edges`` type declaration
``````````````````````````

Note: In the Description column for functions, the relevant init_args are presented as:
``[ comma,seperated,variables | default,values,here ]``

================ =========== =================== ================================================================
 Property         Required    Options             Description                                                    
================ =========== =================== ================================================================
 edges            Y           $NAME               Symbolic identifier for other layers to reference
 type             Y           conv                Layers are fully connected and convolution is applied.
 ..                           max_filter          Layers are connected one-to-one and max filtering is applied.
 init             Y           zero                Filters are zeroed out.
 ..                           constant            Filters are set to a particular constant. ``[ constant | ? ]``
 ..                           uniform             Filters are uniformly randomly initialized. ``[ min,max | -0.1,0.1 ]``
 ..                           gaussian            Filters are gaussian randomly initialized. ``[ mean,stddev | 0,0.01 ]``
 ..                           bernoulli           Filters are bernoulli randomly initialized. ``[ p | 0.5 ]``
 ..                           xavier              Filters are assigned as described in `Glorot and Bengio 2010 <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`_ [1].
 ..                           msra                Filters are assigned as described in `He, Zhang, Ren and Sun 2015 <http://arxiv.org/abs/1502.01852>`_ [2].
 init_args        N           $VALUES             Input comma seperated values of the type appropriate for the selected init.
 size             Y           $X,$Y,$Z            Size of sliding window in pixels. 2D nets can be implemented by setting $Z to 1.
 stride           Y           $X,$Y,$Z            How far to jump in each direction in pixels when sliding the window.
 input            Y           $NODES_NAME         Name of source ``nodes`` layer that the edge will be transforming.
 output           Y           $NODES_NAME         Name of destination ``nodes`` layer that the edge will be transforming.
================ =========== =================== ================================================================

[1] Glorot and Bengio. "Understanding the difficulty of training deep feedforward neural networks". JMLR 2010. http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

[2] He, Zhang, Ren and Sun. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" CVPR 2015. http://arxiv.org/abs/1502.01852
 
For more examples, please refer to the folder ``/opt/znn-release/networks``.

6. Configuration File Format
----------------------------

============================ ========================= ================================================================
 Property                     Options                   Description                                                    
============================ ========================= ================================================================
 fnet_spec                    $ZNN_FILE                 Path to ``.znn`` network architecture file.
 fdata_spec                   $SPEC_FILE                Path to ``.spec`` data description file.
 num_threads                  0..$NUM_CORES             Number of threads to run ZNN on. Bigger is better up to the number of cores you have. 0 will automatically select the maximum.
 dtype                        float32, float64          Sets the numerical precision of the elements within ZNN. Some experiments on 64 bit machines show a 2x speedup with float32. If you change this, you'll need to recompile after setting or unsetting ZNN_USE_FLOATS in the Makefile.
 out_type                     boundary, affinity        Boundary output type is a binary classification, while affinity will give X,Y,Z affinities between neighboring voxels.
 logging                      yes, no                   Record log and config files during your run as a text file. 
 train_outsz                  $Z,$Y,$X (integers)       For each forward pass, this is the size of the output patch.
 cost_fn                      auto                      ``auto`` mode will match the ``out_type``: boundary => softmax, affinity => binomial
 ..                           square_loss               ..
 ..                           binomial_cross_entropy    ..
 ..                           softmax_loss              ..
 eta                          $FLOAT in [0, 1]          Learning rate, η. Controls stochastic gradient descent rate.
 anneal_factor                $FLOAT in [0, 1]          Reduce learning rate by this factor every so often.
 momentum                     $FLOAT in [0, 1]          Resist sudden changes in gradient direction. `More information <https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Neural_Network_Basics#Momentum>`_. 
 weight_decay                 $FLOAT in [0, 1]          A form of regularization, this exponent forces the highest weights to decay. Applied after every iteration.
 Num_iter_per_annealing       $INTEGER                  Number of weight updates before updating ``eta`` by the ``anneal_factor``
 train_net                    $DIRECTORY_PATH           Save intermediate network states into an ``.h5`` file in this directory. Note that ``.h5`` can store more than just image data. If you don't provide a seed (see "Resume a Training" below), this will automatically load.                   
 train_range                  $SAMPLE_NUMBERS           Which samples (defined in your ``.spec``) to train against. You can specify them like 1-3,6 if you wanted to train 1,2,3, and 6.            
 train_conv_mode              fft                       Use FFT for all convolutions.
 ..                           direct                    Use direct convolution all the time.
 ..                           optimize                  Measure and automatically apply FFT or direct per layer based on time performance. Note, this can take several minutes.
 is_data_aug                  yes, no                   Randomly transform patches to enrich training data, including rotation, flipping.
 is_bd_mirror                 yes, no                   In order to provide the sliding window with useful information at the boundaries, mirror the image near the boundaries.
 rebalance_mode               none                      Don't do anything special.
 ..                           global                    Use this when certain classes are disproportionately represented in the training data. This will rebalance the learning process by the global fraction of voxels that each class comprises.
 ..                           patch                     Use this when certain classes are disproportionately represented in the training data. This will rebalance the learning process by the patch fraction of voxels that each class comprises.
 is_malis                     yes, no                   Use Malis for measuring error. c.f. `Turaga, Briggmann, et al. (2009) <http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation>`_ [1]
 malis_norm_type              none                      No normalization
 ..                           frac                      Segment fractional normalization
 ..                           num                       Normalized by N (number of nonboundary voxels)
 ..                           pair                      Normalized by N * (N-1)
 Num_iter_per_show            $INTEGER                  Number of iteration per output.
 Num_iter_per_test            $INTEGER                  Number of iteration per validation/test during training.
 test_num                     $INTEGER                  Number of forward passes of each test.
 Num_iter_per_save            $INTEGER                  Number of iteration per save.
 Max_iter                     $INTEGER                  Maximum iteration limit.
 forward_range                $SAMPLE_NUMBERS           Which samples (defined in your ``.spec``) to run forward against. You can specify them like 1-3,6 if you wanted to train 1,2,3, and 6.            
 forward_net                  $FILE_PATH                ``.h5`` file containing the pre-trained network.
 forward_conv_mode            fft, direct, optimize     Confer ``train_conv_mode`` above.
 forward_outsz                $Z,$Y,$X                  The output size of one forward pass: z,y,x. The larger the faster, limited by the memory capacity.
 output_prefix                $DIRECTORY_PATH           Directory to output the forward pass results.
 is_stdio                     ..                        `Standard IO format <https://docs.google.com/spreadsheets/d/1Frn-VH4VatqpwV96BTWSrtMQV0-9ej9soy6HXHgxWtc/edit?usp=sharing>`_ in Seunglab. If yes, will 
 ..                           yes                       Save the learning curve and network in one file. (recommended for new training)
 ..                           no                        For backwards compatibility, save learning curve and network in seperate files.
 is_debug                     yes, no                   Output some internal information and save patches in network file.
 is_check                     yes, no                   Check the patches, used in Travis-ci for automatic test
============================ ========================= ================================================================

[1] Turaga, Briggmann, et al. "Maximin affinity learning of image segmentation". NIPS 2009. http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation

5. TO DO
-----------
- Publicly available ZNN AWS AMI
- Describe all the code in plain English using comments
- Do we want to add in AWS tutorial too (see http://cs224d.stanford.edu/supplementary/aws-tutorial-2.pdf)?
- State which instance type to use
- Be clearer about output size parameter and effect on memory
