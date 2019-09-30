from __future__ import absolute_import, division, print_function
from google.protobuf import text_format
import tensorflow as tf
import tensorflow_datasets as tfds
import time

tfds.disable_progress_bar()
# tf.compat.v1.disable_eager_execution()

CLUSTER_SPEC = {"worker": ["localhost:1111", "localhost:1112", "localhost:1113"]}


def start_server(job_name, task_index, tf_config):
    """ Create a server based on a cluster spec. """
    cluster = tf.train.ClusterSpec(CLUSTER_SPEC)
    server = tf.compat.v1.train.Server(
        cluster, config=tf_config, job_name=job_name, task_index=task_index)
    return server, cluster


job_name = "worker"
task_index = 0
# Set up tensorflow configuration.
tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# Start a server.
server, cluster = start_server(job_name, task_index, tf_config)

a = tf.Variable(2.0, name='a')

if job_name == "ps":
    server.join()
else:
    with tf.compat.v1.Session(server.target) as sess:
        a = tf.Variable(tf.ones([6, 6]), name='a')
        sess.run(tf.compat.v1.global_variables_initializer())
        print(sess.run(a))
        time.sleep(10)
        exit(0)
