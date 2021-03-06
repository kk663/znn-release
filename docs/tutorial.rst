ZNN + AWS Tutorial
==================

The tutorial will help you learn how to use ZNN with the help of AWS (Amazon Web Services). We have an AMI (Amazon Machine Image) with ZNN pre-installed so that you can just launch an AWS EC2 (Elastic Compute Cloud) instance using the AMI and run the tutorial on the instance (no need to deal with installation issues). The goal of the tutorial is to train a 2D sliding-window CNN (Convolutional Neural Network) to perform boundary detection. In particular, the tutorial will focus on the training of the N4 network described in the paper `"Deep Neural Networks Segment Neuronal Membranes in Electron Microscopy Images" <https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images>`_.

The tutorial focuses on usage of the python interface since it is more convenient to use.

The tutorial assumes that you are already familiar with how to use AWS. If you are not familiar with how to use AWS, please consult the AWS tutorial `here <https://cs224d.stanford.edu/supplementary/aws-tutorial-2.pdf>`_. The ZNN AMI is publicly available in the ``US East (N. Virginia)`` region under the name ``ZNN Public Release`` with AMI ID ``ami-161d9101``.

The tutorial also assumes that you are somewhat familiar with neural networks and how to train them.

Please note that you can run ZNN on persistent spot instances in AWS EC2 with the help of starcluster. This helps reduce the cost of using AWS. We have prepared a `README <https://github.com/seung-lab/znn-release/blob/master/python/aws_train/README.md>`_ with instructions on how to use ZNN with starcluster.

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

The dataset is already provided in the AMI. It can be found in the folder ``/opt/znn-release/dataset/test``. This folder should contain the following files: ``dataset.spec``, ``stack1-label.tif``, ``stack2-label.tif``, ``stack1-image.tif`` and ``stack2-image.tif``.

Image configuration
```````````````````

ZNN requires a ``.spec`` file that provides the binding between the raw image stacks and the labelled image stacks (i.e. ground truth) in the dataset.

The raw image stack and the corresponding labelled image stack are defined as a **Sample**.

The ``.spec`` file format allows you to specify multiple files as inputs (stack images) and outputs (ground truth labels) for a given experiment (you can have multiple input stacks for both training and inference or forward pass).

Recall that we have two image stacks: stack1 and stack2. Let's define the raw image stacks as inputs in the ``.spec`` file:
::
    [image1]
    fnames = ../dataset/test/stack1-image.tif
    pp_types = standard2D
    is_auto_crop = yes
    
    [image2]
    fnames = ../dataset/test/stack2-image.tif
    pp_types = standard2D
    is_auto_crop = yes

We must use ``[image]`` sections to indicate the network inputs. For stack1, we must first include the ``[image1]`` header. Note that the number in the header (at the end) can be any positive integer. Next, we must specify the full file path of raw image stack in the field ``fnames``. The ``pp_types`` or preprocessing types field does not need to be set. It is ``none`` by default. We set it to be ``standard2D`` so that we subtract the mean and divide by the standard deviation of the pixel values for each 2D image in the raw image stack. The field ``is_auto_crop`` is set to ``no`` by default. We set the ``is_auto_crop`` field to ``yes`` so if the raw image is of different size than the ground truth image, the smaller image is centered in the larger image and the larger image is cropped around the smaller image. We then do the same thing (section header, full file path field, preprocessing types field and autocrop field) for stack2.

Now let's define the labelled image stacks as ground truth outputs in the ``.spec`` file:
::
    [label1]
    fnames = ../dataset/test/stack1-label.tif
    pp_types = binary_class
    is_auto_crop = yes
    fmasks =
    
    [label2]
    fnames = ../dataset/test/stack2-label.tif
    pp_types = binary_class
    is_auto_crop = yes
    fmasks =

We must use ``[label]`` sections to indicate the ground truth of network outputs. For stack1, we must first include the ``[label1]`` header. Note that the number in the header (at the end) can be any positive integer. Next, we must specify the full file path of labelled image stack in the field ``fnames``. The ``pp_types`` or preprocessing types field does not need to be set. It is ``none`` by default. We set it to be ``binary_class`` so that we generate two 2D output images for each 2D input image. The first 2D output image is the original labelled 2D image and the second 2D output image is the inverted version of the original labelled 2D image. The field ``is_auto_crop`` is set to ``no`` by default. We set the ``is_auto_crop`` field to ``yes`` so if the raw image is of different size than the groundtruth image, the smaller image is centered in the larger image and the larger image is cropped around the smaller image. The ``fmasks`` field is for full file paths of masks: fmasks are used like cosmetics to coverup damaged parts of images so that your neural net doesn’t learn useless information. We have no damaged image parts in our dataset so we do not need to specify the full file paths of masks. We then do the same thing (section header, full file path field, preprocessing types field, autocrop field and full file path of masks field) for stack2.

Next, let's define the bindings of raw image stacks and the corresponding labelled image stacks in the ``.spec`` file:
::
    [sample1]
    input = 1
    output = 1
    
    [sample2]
    input = 2
    output = 2

We must use ``[sample]`` sections to indicate the pairing of the raw image stacks and the corresponding labelled image stacks (each sample can be thought of as a raw image stack and the corresponding labelled image stack). For stack1, we must first include the ``[sample1]`` header. Note that the number in the header (at the end) can be any positive integer. Next, we must specify that the input is the raw image stack with section header ``image1`` and the (ground truth) output is the labelled image stack with section header ``label1``. We then repeat the same thing (section header, input field and output field) for stack2.

The full code can be found in the ``dataset.spec`` file in the folder ``/opt/znn-release/dataset/test`` and is as follows:
::
    # samples example
    # the [image] sections indicate the network inputs
    # format should be gray images with any bit depth.
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
    # only the voxels with gray value greater than 0 are effective for training.
    #
    # [label1]
    # fnames = path/of/image3.tif/h5,
    #          path/of/image4.tif/h5
    # preprocessing type: one_class, binary_class, none, affinity
    # pp_types = binary_class, binary_class
    # fmasks = path/of/mask1.tif/h5,
    #      path/of/mask2.tif/h5
    #
    # [sample] section indicates the group of the corresponding input and output labels
    #
    # [sample1]
    # input1 = 1
    # input2 = 2
    # output1 = 1
    # output2 = 2
    
    [image1]
    fnames = ../dataset/test/stack1-image.tif
    pp_types = standard2D
    is_auto_crop = yes
    
    [image2]
    fnames = ../dataset/test/stack2-image.tif
    pp_types = standard2D
    is_auto_crop = yes
    
    [label1]
    fnames = ../dataset/test/stack1-label.tif
    pp_types = binary_class
    is_auto_crop = yes
    fmasks =
    
    [label2]
    fnames = ../dataset/test/stack2-label.tif
    pp_types = binary_class
    is_auto_crop = yes
    fmasks =
    
    [sample1]
    input = 1
    output = 1
    
    [sample2]
    input = 2
    output = 2


2. Network Architecture Configuration
-------------------------------------

We have a custom file format ``.znn`` for specifying the layout of your neural network. It works based on a few simple concepts. 

1. Each of the input nodes of the network represents an image stack.
2. The network consists of layers whose size can be individually specified. 
3. The edges between the layers specify not only the data transfer from one layer to another (e.g. one to one or fully connected), they also prescribe a transformation (e.g. a filter or weight) to be applied. 
4. After all the weights or filters have been applied, the inputs are summed and a pixel-wise transfer function (e.g. a `sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function>`_ or `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_) is applied.
5. The type of the edges determines if the layers connected have a one-to-one mapping or are fully connected. For example, a convolution type will result in fully connected layers.
6. The output layer represents whatever you are training the network to do. One common output is the predicted labels for an image stack as a single node.

We shall now define the network architecture of the N4 net. Let's start with defining the input layer:
::
    nodes input
    type input
    size 1 

The command ``nodes layer-name`` is used to declare a layer with name ``layer-name``. First, we declare that the layer is the input layer using the command ``nodes input``. Note that the ``layer-name`` of ``input`` is special and is reserved for the input layer. Then we specify that the layer is of ``type input``. Next, the command ``size 1`` indicates that there is only one feature map in the input layer (the input stacks contain 2D grayscale images so there is only one image channel).

We would like the next layer to be a convolutional layer. We must define the edges between the input layer and the next layer before defining the next layer:
::
    edges conv1
    type conv
    init xavier
    size 1,4,4
    stride 1,1,1
    input input
    output nconv1

The command ``edges edges-name`` is used to declare edges with collective name ``edges-name``. First, we declare the edges with name ``conv1`` using the command ``edges conv1``. Then we specify that the layers are to be fully-connected and convolution is applied with command ``type conv``. The command ``init xavier`` specifies that the weights on the edges are to be initialized using `Xavier initialization <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`_. The size of the convolutional kernel is ``4 x 4 x 1`` (x, y, z dimensions) with stride ``1`` in all dimensions. These is specified using the commands ``size 1,4,4`` and ``stride 1,1,1`` respectively. Next, we specify the input ``nodes`` layer or source layer (layer from which the edges originate) and the output ``nodes`` layer or destination layer (layer to which the edges travel) using the commands ``input input`` and ``output nconv1``. Observe that we used the layer-name we gave to the input layer (``input``) and we must declare the layer-name that we will give to the next layer (``nconv1``) in the edges section that precedes the declaration of the next layer.

After defining the edges between the input layer and the next layer, we must now define the next layer:
::
    nodes nconv1
    type transfer
    function rectify_linear
    size 48

The command ``nodes nconv1`` defines the name of the layer to be ``nconv1`` and this name must match the one used for the next layer in the preceding edge section. Then we specify that the layer is of type ``transfer`` using command ``type transfer``. This means that the layer will take the values stored in the input neurons (in the source layer of the edges object declared above) and compute a weighted sum at each output neuron (in the destination layer of the edges object declared above, which is this layer) using the weights on the edges. Finally, the layer will apply a transfer function such as sigmoid or ReLU to the weighted sum stored in each output neuron. Next, we specify the transfer function of the layer to be ``rectify_linear`` (this is the ReLU activation function) using the command ``function rectify_linear``. Finally, we must specify the number of nodes or feature maps in this layer (which must be a positive integer). We set the number of feature maps in this layer to be 48 using the command ``size 48``.

We then wish to apply max-pooling to significantly increase the 2D receptive field of our network and make our network more invariant to minor translations:
::
    edges pool1
    type max_filter
    size 1,2,2
    stride 1,2,2
    input nconv1
    output npool1
    
    nodes npool1
    type sum
    size 48

We first define the edges that go from the previous layer to the pooling layer using command ``edges pool1``. We then specify that the edges object type is to be ``max_filter`` using command ``type max_filter``. This just means that the layers are connected one-to-one and max-filtering (i.e. max-pooling) is applied. The size of the pooling filer is specified to be ``2 x 2 x 1`` (x, y, z dimensions) with stride 1 in the z-dimension, stride 2 in the y-dimension and stride 2 in the x-dimension using commands ``size 1,2,2`` and ``stride 1,2,2``. We then define the input and output layers of the edges object.

Afterwards, we must define the pooling layer nodes object using command ``nodes npool1``. We specify that the nodes object type is ``sum`` using command ``type sum`` and that there are 48 output feature maps using command ``size 48``. In this context, the ``sum`` type means that the values in the previous layer will be "max-pooled" and then transmitted to the next layer.

We can then define the rest of the network using similar ``node`` and ``edge`` object declarations. Special attention must be paid to the declaration of the last layer of the network:
::
    nodes output
    type transfer
    function linear
    size 2

The last layer of the network is declared to have special name ``output`` using the command ``nodes output``. We specify that the transfer function (i.e. activation function) is to be ``linear`` (i.e. ``y = ax + b``) and this is to be applied to the weighted sum at each neuron in the last layer using commands ``type transfer`` and ``function linear``. We are trying to predict the neuronal boundaries by considering each pixel in the stack individually. We use a sliding window (with mirroring at the boundaries so no zero-padding is applied) that takes a fixed-size context image around the pixel-to-be-classified. For each pixel, we must output the probability of it being membrane and the probability of it being non-membrane. We thus must output 2 feature maps, one which contains the membrane pixel probabilities and the other which contains the non-membrane pixel probabilities. We specify that 2 feature maps must be output (one for each class) using the command ``size 2``.

Please note that all forward pass convolutions are of type valid while all backpropagated convolutions are of type full. Furthermore, the size of the sliding window or context window is automatically determined by ZNN. ZNN computes the field-of-view of the stacked convolutional layers (with pooling) and sets the size of the context window to be equal to that of the field-of-view. In addition, the max-pooling operation is carried out in a manner similar to the valid forward pass convolutions.

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

The ``.znn`` file is comprised of two primary objects -- nodes and edges. An object declaration consists of the type ``nodes`` or ``edges`` followed by its name on the same line and then followed by its parameters.

3. Training
-----------

Now that you've set up your training and validation datasets in your ``.spec`` file and have designed a neural network in your ``.znn`` file, 
it's time to tell the network exactly what to do. We do this via a ``.cfg`` configuration file.

Parameter configuration
```````````````````````
The training and forward parameters of the network can be set using a configuration file. 

The configuration file uses the commonly used `Python ConfigParser <https://docs.python.org/2/library/configparser.html>`_. Consult that link for detailed information on acceptable syntax.
The ``.cfg`` file uses ``[sections]`` to ecapsulate different parameter sets. In the past, we used to use multiple sections, but now we just use one called ``[parameters]``.

First, we must specify the full file path of the network architecture or ``.znn`` file:
::
    fnet_spec = ../networks/N4.znn

Then we must specify the full file path of the dataset specification file or ``.spec`` file:
::
    fdata_spec = ../dataset/test/dataset.spec

We next specify the type of network output, which can be either `boundary <https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images>`_ or `affinity <http://web.mit.edu/people/sturaga/papers/Turaga2010NC.pdf>`_:
::
    out_type = boundary

During training, ZNN will periodically save the learned network in a file (e.g.: net_21000.h5, net_current.h5). We must specify the prefix file path for these files:
::
    train_net_prefix = ../experiments/piriform/N4/net

Next, we must specify the sample ID numbers for the train and validation/test stacks. In this tutorial, we use stack2 for training and stack1 for testing:
::
    # example: 2-3,7
    train_range = 2
    test_range = 1

ZNN performs optimization using a procedure similar to mini-batch stochastic gradient descent. Recall that we are classifying each pixel in the 2D image slices by taking a context window centered at that pixel and feeding that window to a CNN ("sliding window" approach). To speed things up during training, ZNN takes several context windows in a region, performs a forward pass on them and then updates the weights of the CNN model (this is like mini-batch stochastic gradient descent but not quite because the context windows overlap). Note that this step is referred to as an update/iteration in the ZNN command line output. We call this dense output forward pass and it speeds up training significantly. However, it comes at the cost of high memory usage. The code snippet below shows an example for setting the train dense output size (the train dense output size can be as big as the size of a single 2D image slice for 2D sliding window CNNs but a train dense output size of something like ``5, 70, 70`` is recommended for 3D sliding window CNNs):
::
    # dense output size of one forward pass: z,y,x
    # large output size can reduce the computational redundency
    # this parameter affects the memory consumption a lot.
    # keep an eye to the memory, if it occupies too much memory, reduce this outsz
    train_outsz = 1,100,100

Afterwards, we must specify the hyperparameters for the model-to-be-learned:
::
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

Then, we must specify the data augmentation, mirroring and training data balance properties for training:
::
    # randomly transform patches to enrich training data, including rotation, fliping
    is_data_aug = yes
    # mirror the image region close to boundaries to get a full size output
    is_bd_mirror = yes
    # balance the boundary and non-boundary voxel
    # global: compute the weight in the whole image stack
    # patch: compute the balance weight for each patch
    rebalance_mode = global

Then, we must specify the command line output during training (recall that one iteration/update is simply one "mini-batch" stochastic gradient descent step):
::
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

Training needs to be manually monitored as it proceeds (i.e. read the ZNN command line output). Look carefully at the rand error and pixel error. If the validation pixel error flat-lines over many iterations/updates, try stopping training and then resuming training with smaller learning rate ``eta``. If the validation pixel error goes up over many iterations/updates, stop training immediately to prevent overfitting. Unless training is terminated manually, it will continue until the maximum number of iterations is reached as specified by field ``Max_iter``.

Finally, we must specify the forward pass or inference properties. We use the test stack or stack1 for performing forward pass as specified by ``forward_range``. The forward pass is performed using the full file path of the trained model specified by ``forward_net``. Similar to training, we can perform dense forward pass output to speed up inference on a test stack at the cost of memory by specifying the field ``forward_outsz``. The output of ZNN forward pass is a ``.h5`` file with the boundary detection output for each class or one ``.tif`` file containing the boundary detections for each class. The output prefix file path is specified using the field ``output_prefix``:
::
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

Make sure you run the following command from within the ``/opt/znn-release/python`` directory. This is a limitation that can be fixed in future releases.
::
    python train.py -c config.cfg 

NOTE: If your training aborts without writing any iterations/updates, try reducing the output size as you may have run out of memory.

Resume training the N4 network
``````````````````````````````
Since the network is periodically saved, we can resume training whenever we want to. By default, ZNN will automatically resume the latest training net (``net_current.h5``) in a folder, which was specified by the ``train_net`` parameter in the configuration file. 

To resume training a specific network, we can use the seeding function:
::
    python train.py -c config.cfg -s path/of/seed.h5

Transfer learning using the N4 network
``````````````````````````````````````
Sometimes, we would like to utilize a trained network. If the network architectures of trained and initialized network are the same, we call it ``Loading``. Otherwise, we call it ``Seeding``, in which case the trained net is used as a seed to initialize part of the new network. Our implementation merges ``Loading`` and ``Seeding``. Just use the synonymous ``-s`` or ``--seed`` command line flags. 
::
    python train.py -c config.cfg -s path/of/seed.h5

Forward Pass using the N4 network
`````````````````````````````````
run the following command:
::
    python forward.py -c config.cfg
if you are running forward pass intensively for a large image stack, it is recommanded to recompile python core using `DZNN_DONT_CACHE_FFTS`. Without caching FFTS, you can use a large output size, which reuse a lot of computation and speed up your forward pass.

NOTE: If your forward pass aborts without writing anything, try reducing the output size as you may have run out of memory.

4. Instructions for Running Tutorial Code
-----------------------------------------

Step 1 - Launch AWS EC2 instance of type ``c4.8xlarge`` (with 30 GB of EBS or Elastic Block Store storage) using the ZNN AMI. To find the ZNN AMI, set your AWS region to ``US East (N. Virginia)``, select the ``Community AMIs`` tab and search for ``ZNN``. You should find the ZNN AMI under the name ``ZNN Public Release - ami-161d9101`` with description ``Public AMI for running ZNN out-of-the-box. Comes with segascorus error-scoring software``.

Step 2 - ssh into launched instance via the command line

Step 3 - Enter the commands below (note that ZNN should state that the field of view or sliding window size of the N4 network is ``(1, 95, 95)``, which is ``95 x 95 x 1`` in the x, y, z dimensions):
::
    sudo su
    cd /opt/znn-release/python
    screen
    python train.py -c config.cfg

Step 4 - Monitor how the training of the neural network proceeds.  You should see output similar to the following as training proceeds:
::
    start from  1
    save as  ../experiments/piriform/N4/net_init.h5
    stdpre:  /processing/znn/train/statistics/
    update 100,    cost: 0.341, pixel error: 0.465, rand error: 0.198, elapsed: 0.7 s/iter, learning rate: 0.010
    update 200,    cost: 0.301, pixel error: 0.347, rand error: 0.217, elapsed: 0.7 s/iter, learning rate: 0.010
    test iter: 200,     cost: 0.303, pixel error: 0.350
    update 300,    cost: 0.279, pixel error: 0.309, rand error: 0.193, elapsed: 0.7 s/iter, learning rate: 0.010
    update 400,    cost: 0.252, pixel error: 0.269, rand error: 0.178, elapsed: 0.7 s/iter, learning rate: 0.010
    test iter: 400,     cost: 0.273, pixel error: 0.301
    update 500,    cost: 0.234, pixel error: 0.236, rand error: 0.155, elapsed: 0.7 s/iter, learning rate: 0.010
    update 600,    cost: 0.242, pixel error: 0.248, rand error: 0.163, elapsed: 0.7 s/iter, learning rate: 0.010
    test iter: 600,     cost: 0.246, pixel error: 0.255
    update 700,    cost: 0.239, pixel error: 0.245, rand error: 0.162, elapsed: 0.7 s/iter, learning rate: 0.010
    update 800,    cost: 0.233, pixel error: 0.236, rand error: 0.157, elapsed: 0.7 s/iter, learning rate: 0.010
    test iter: 800,     cost: 0.223, pixel error: 0.226
    update 900,    cost: 0.206, pixel error: 0.197, rand error: 0.136, elapsed: 0.7 s/iter, learning rate: 0.010
    update 1000,    cost: 0.209, pixel error: 0.202, rand error: 0.152, elapsed: 0.7 s/iter, learning rate: 0.010
    test iter: 1000,     cost: 0.212, pixel error: 0.213
    save as  ../experiments/piriform/N4/net_1000.h5
    stdpre:  /processing/znn/train/statistics/
    update 1100,    cost: 0.214, pixel error: 0.206, rand error: 0.162, elapsed: 0.7 s/iter, learning rate: 0.010
    update 1200,    cost: 0.204, pixel error: 0.198, rand error: 0.152, elapsed: 0.7 s/iter, learning rate: 0.010
    test iter: 1200,     cost: 0.238, pixel error: 0.248

ZNN does not check convergence and so it will run until the maximum number of iterations specified in the configuration file. To detach the window (using screen), simply type ``ctrl + A + D``. When you ssh back into your instance, just type the commands below and you will be able to see how training is progressing:
::
    sudo su
    cd /opt/znn-release/python
    screen -r

Step 5 - Once you have determined that the neural network fits the data well enough (e.g.: flat-line in rand score), simply terminate training by typing ``ctrl + C``. Try training the neural network for 2-3 hours before terminating training.

Step 6 - We now need to run inference or forward pass on the test stack (stack1) using the trained neural network model (note that we train on stack2). Type the following commands (need to be root user - type ``sudo su`` if needed):
::
    cd /opt/znn-release/python
    python forward.py -c config.cfg

Step 7 - Enter the commands below to evaluate the performance of the neural network on the test stack (need to be root user - type ``sudo su`` if needed):
::
    cd /opt/znn-release/segascorus
    python error.py /opt/znn-release/experiments/piriform/N4/out_sample1_output_0.tif /opt/znn-release/dataset/test/stack1-label.tif

Step 8 - The python script should output something similar to the content below:
::
    Rand Error Full: 0.0373468767395
    Rand Error Merge: 6.46534700016e-06
    Rand Error Split: 0.0373404113925
    Rand F Score Full: 9.34609309984e-05
    Rand F Score Merge: 0.212575061507
    Rand F Score Split: 4.67407405359e-05
    Variation F Score Full: 0.398724422948
    Variation F Score Merge: 0.998632619193
    Variation F Score Split: 0.249089176191
    Variation of Information Full: 11.967491642
    Variation of Information Merge: 0.00543320565003
    Variation of Information Split: 11.9620584363

Please see the document `here <https://github.com/seung-lab/segascorus/blob/master/segerror-manual.pdf>`_ for more information on the metrics used above.
