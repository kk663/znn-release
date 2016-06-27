ZNN Training on AWS using Spot Instances
=======================================
This script can create a cluster including an on-demand master node and several spot-instance worker nodes. Whenever the spot instance node gets terminated by price, the script will create a new spot instance request. Thus the script creates a kind of "persistent" spot worker node.

##Setup
* [Install StarCluster](http://star.mit.edu/cluster/docs/latest/installation.html). `sudo easy_install StarCluster`. If using Mac OS, you will need to install Homebrew and then use it to install OpenSSL.
  * Enter ``/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`` in terminal to install Homebrew (Mac OS only).
  * Enter ``brew install openssl`` in terminal to install OpenSSL (Mac OS only).
  * Enter ``brew link openssl --force`` in terminal to sym-link OpenSSL (Mac OS only).
* [Download StarCluster](https://github.com/jtriley/StarCluster) and set the StarCluster folder path on your machine as the PYTHONPATH.
  * Enter ``git clone https://github.com/jtriley/StarCluster.git`` in terminal
  * Put the line `export PYTHONPATH=$PYTHONPATH:"/path/to/StarCluster"` at the end of `~/.bashrc` file on Linux or at the end of `~/.bash_profile` file on Mac OS.
  * Run `source ~/.bashrc` in terminal if using Linux.
  * Exit all terminals. Create a new terminal.
* Edit and move `config` file to `~/.starcluster/`.
  * Setup the keys in `config`. Please see [AWS Crediential Tutorial](http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html) for information on how to set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
  * Set the AMI ID. The AMI ID should be set as follows: `NODE_IMAGE_ID = ami-23a4454e`.
  * `AWS_USER_ID` should be set to your AWS account ID (see http://docs.aws.amazon.com/general/latest/gr/acct-identifiers.html).
  * Under heading "Defining EC2 Keypairs", please set the `XXX` in `[key XXX]` to be the name of your AWS EC2 key pair and `KEY_LOCATION` to be the full file path of the `.pem` file corresponding to the key pair on your computer (please see http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html for more information on key pairs).
  * Under heading "Defining Cluster Templates", set the `KEYNAME` field to be the name of the AWS EC2 key pair used above.
* Copy the `train_example.cfg` file as `train.cfg`
* Set some additional parameters in the `train.cfg`.
    * Cluster name
    * Node name (Note that do not use `_`!)
    * Instance type
    * Bidding for the spot instance
    * Commands for each spot instance
* Create a volume using starcluster (it won't work for volume created in web console!): enter `starcluster createvolume 50 us-east-1c` into a terminal. This will create an EBS volume of size 50 GB on AWS EC2 in availability `region us-east-1c`. You can get a volume ID from this step (see the AWS EC2 console to get the volume ID). This volume will be your persistent storage for all the training files.
* Edit the `config` file in `~/.starcluster/` so that `VOLUME_ID` is set to the volume ID assigned in the previous step

##Tutorial
* Create a volume using starcluster (it won't work for volume created in web console!): `starcluster createvolume 50 us-east-1c`, you can get a volume ID from this step. This volume will be your persistent storage for all the training files.
* Check your cluster: `starcluster listclusters`
* Terminate your volume-creator cluster by: `starcluster terminate -f volumecreator`
* Setup the volume id in starcluster configure file.
* Launch a cluster only has the `master`: `starcluster start mycluster`
* Set the `node_name` in script to choose the command you want to run. (normally, we use network name as node name)
* Modify the command dict to execute training commands after the node was launched. the `node_name` is the key of command dict.

##Run
* run the main script: `python aws_train.py mynode`, `mynode` is the node name
* use `starcluster sshnode mycluster mynode` to login your node. 
* go to the persistent volume: `cd /home`
* start training and have fun!

##Usage
* Check your cluster: `starcluster listclusters`
* ssh: `starcluster sshmaster mycluster`
* upload: `starcluster put mycluster myfile clusterfile`
* download: `starcluster get mycluster clusterfile myfile`
* get help: `starcluster help`
