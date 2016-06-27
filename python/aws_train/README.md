ZNN Training on AWS using Spot Instances
=======================================
This script can create a cluster including an on-demand master node and several spot-instance worker nodes. Whenever the spot instance node gets terminated by price, the script will create a new spot instance request. Thus the script creates a kind of "persistent" spot worker node.

##Setup
* [Install StarCluster](http://star.mit.edu/cluster/docs/latest/installation.html). `sudo easy_install StarCluster`. If using Mac OS, you will need to install Homebrew and then use it to install OpenSSL.
  * Enter ``/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`` in terminal to install Homebrew (Mac OS only)
  * Enter ``brew install openssl`` in terminal to install OpenSSL (Mac OS only)
  * Enter ``brew link openssl --force`` in terminal to sym-link OpenSSL (Mac OS only)
* [Download StarCluster](https://github.com/jtriley/StarCluster) and set the StarCluster folder path on your machine as the PYTHONPATH.
  * Enter ``git clone https://github.com/jtriley/StarCluster.git`` in terminal
  * Put the line `export PYTHONPATH=$PYTHONPATH:"/path/to/StarCluster"` at the end of `~/.bashrc` file on Linux or at the end of `~/.bash_profile` file on Mac OS.
  * Run `source ~/.bashrc` in terminal if using Linux.
  * Exit all terminals. Create a new terminal.
* Edit and move `config` file to `~/.starcluster/`.
  * Setup the keys in `config`.
  * Set the AMI and volume id.
  * Setup all the parameters with a mark of `XXX`
* copy the `train_example.cfg` file as `train.cfg`
* set some additional parameters in the `train.cfg`.
    * cluster name
    * node name (Note that do not use `_`!)
    * instance type
    * biding for the spot instance
    * commands for each spot instance

##Tutorial
now, you are almost ready. 
* create a volume using starcluster (it won't work for volume created in web console!): `starcluster createvolume 50 us-east-1c`, you can get a volume ID from this step. This volume will be your persistent storage for all the training files.
* check your cluster: `starcluster listclusters`
* terminate your volume-creator cluster by: `starcluster terminate -f volumecreator`
* setup the volume id in starcluster configure file.
* launch a cluster only has the `master`: `starcluster start mycluster`
* set the `node_name` in script to choose the command you want to run. (normally, we use network name as node name)
* modify the command dict to execute training commands after the node was launched. the `node_name` is the key of command dict.

# run
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
