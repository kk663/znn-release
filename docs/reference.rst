ZNN Reference
=============

1. Specification File Format
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

2. ZNN File Format
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

3. Configuration File Format
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
