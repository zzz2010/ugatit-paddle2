# coding=utf-8
"""
Welcome to AI Studio. Let us get familiar with the rules of AI Studio Script Tasks before use it.
"""

# Directory of Dataset files
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'

# The specific path of the dataset file can be obtained through the file 'Path Copy' button in the 'Dataset' tab  in the left navigation bar.
train_datasets =  '通过路径拷贝获取真实数据集文件路径 '

# Directory of Output files. After task completed, all results will be packed into a tar.gz file and put it in  this directory. You can download the output file to the local via "Download Output" link.
output_dir = "/root/paddlejob/workspace/output/"

# Log record.
# The task will automatically record the log about  environment initialization, task execution, error, all standard output and standard error flow in the execution (such as print()). You can track log information by 'View Log' after 'Submit Task' .

from argparse import ArgumentParser, REMAINDER
import paddle.distributed.launch as launch

from argparse import Namespace


# launch args
parser = ArgumentParser()
parser.add_argument("--cluster_node_ips", type=str, default="127.0.0.1",
    help='# Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..')
parser.add_argument("--node_ip", type=str, default="127.0.0.1", help='The current node ip.')
parser.add_argument("--use_paddlecloud", action='store_true', default=False,
    help='wheter to use paddlecloud platform to run your multi-process job. If false, no need to set this argument.')
parser.add_argument("--started_port", type=int, default=None, help="The trainer's started port on a single node")
parser.add_argument("--print_config", type=bool, default=True, help='Print the config or not')
parser.add_argument("--selected_gpus", type=str, default=None,
    help="It's for gpu training and the training process will run on the selected_gpus, each "
         " process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training.")
parser.add_argument("--log_level", type=int, default=20, help='Logging level, default is logging.INFO')
parser.add_argument("--log_dir", type=str, default=None,
    help="The path for each process's log.If it's not set, the log will printed to default pipe.")

args = parser.parse_args()

args.training_script="改成自己的训练启动py文件，比如train.py"
args.training_script_args=["--dataset_base_path", datasets_prefix + "data65/", "--output_base_path", output_dir + "model"]

launch.launch(args)
