

import sonnet as snt
import tensorflow as tf
from dataloader import *
from models import *
import os


checkpoint_root = "./checkpoints"
checkpoint_name = "model"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)

graph_network = Captcha()
checkpoint = tf.train.Checkpoint(module=graph_network)

latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
  checkpoint.restore(latest)

code = Code("../pytorch_verification_code/dataset/train",4,8,56,100)
max_iteration = 1000000
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

for echo in range(max_iteration):
	code.mess_up_order()
	
	for i in range(code.total_number):
		with tf.GradientTape() as gen_tape:
			nodes,edges_index,edges_attr,u,batch,target = code.next_batch(i)
			x, edge_attr, output = graph_network(nodes, edges_index, edges_attr, u, batch)
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
			
		gradients_of_generator = gen_tape.gradient(loss, graph_network.trainable_variables)
		generator_optimizer.apply_gradients(zip(gradients_of_generator, graph_network.trainable_variables))

		print('Echo %d,Iter [%d/%d]: train_loss is: %.5f train_accuracy is: %.5f'%(echo+1,i+1,code.total_number,tf.reduce_mean(loss),train_accuracy(target,output)))
		
			
		if i and i % 1000 == 0:
			checkpoint.save(save_prefix)
			
checkpoint.save(save_prefix)





		


