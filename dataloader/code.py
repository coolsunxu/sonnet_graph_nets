

import tensorflow as tf

import numpy as np
from pathlib import Path
import cv2
import random

class Code:
	def __init__(self,path_kind,batch_size,stride,img_width,img_height):
		self.alpha = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a','b',
					  'c','d','e','f','g','h','i','j','k','l','m','n','o','p',
					  'q','r','s','t','u','v','w','x','y','z']
					  
		self.data_root = Path(path_kind)
		self.batch_size = batch_size
		self.img_height = img_height
		self.img_width = img_width
		self.stride = stride
		
		self.block_height = int(img_height / stride)
		self.block_width = int(img_width / stride)
		
		self.load()
		
	def load(self):
		
		self.second_image_paths = list(self.data_root.glob('*'))
		self.second_image_paths=[str(path) for path in self.second_image_paths]
		self.total_number = len(self.second_image_paths) // self.batch_size
	
	
	def mess_up_order(self):
		random.shuffle(self.second_image_paths)
	
	def List2Tensor(self,x):
		return tf.reshape(x, [tf.shape(x)[0], -1])
		
	def next_batch(self,index):
		nodes = []
		edges_index = [[],[]]
		edges_attr = []
		u = []
		labels = []
		batch = []
		back_node_num = 0
		for k,path in enumerate(self.second_image_paths[self.batch_size*index:self.batch_size*(index+1)]) : 
			now_node_num = self.block_height * self.block_width
			back_node_num = back_node_num + now_node_num
			temp_str = path.split('/')[-1]
			begin=temp_str.find('_')
			end=temp_str.find('.')
			label = self.alpha.index(temp_str[begin+2:end])
			
			image = cv2.imread(path)
			image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			
			image = ((image / 255.) - .5) * 2.
			
			u_temp = []
			
			batch = batch +[k]*now_node_num
			for i in range(self.block_height):
				for j in range(self.block_width):
					#nodes
					temp_nodes = image[i*self.stride:(i+1)*self.stride,j*self.stride:(j+1)*self.stride].flatten()
					u_temp.append(np.mean(temp_nodes))
					nodes.append(temp_nodes.tolist())
					
					#edges
					if i-1>=0 and j>=0:
						edges_index[0].append(back_node_num-now_node_num+i*self.block_width+j)
						edges_index[1].append(back_node_num-now_node_num+(i-1)*self.block_width+j)
						edges_attr.append(np.array([1,1]).tolist())
						
						edges_index[1].append(back_node_num-now_node_num+i*self.block_width+j)
						edges_index[0].append(back_node_num-now_node_num+(i-1)*self.block_width+j)
						edges_attr.append(np.array([1,1]).tolist())
						
						
					if i+1>=0 and j>=0 and i+1<=self.block_height-1:
						edges_index[0].append(back_node_num-now_node_num+i*self.block_width+j)
						edges_index[1].append(back_node_num-now_node_num+(i+1)*self.block_width+j)
						edges_attr.append(np.array([1,1]).tolist())
						
						edges_index[1].append(back_node_num-now_node_num+i*self.block_width+j)
						edges_index[0].append(back_node_num-now_node_num+(i+1)*self.block_width+j)
						edges_attr.append(np.array([1,1]).tolist())
						
					if i>=0 and j-1>=0:
						edges_index[0].append(back_node_num-now_node_num+i*self.block_width+j)
						edges_index[1].append(back_node_num-now_node_num+i*self.block_width+j-1)
						edges_attr.append(np.array([1,1]).tolist())
						
						edges_index[1].append(back_node_num-now_node_num+i*self.block_width+j)
						edges_index[0].append(back_node_num-now_node_num+i*self.block_width+j-1)
						edges_attr.append(np.array([1,1]).tolist())
						
					if i>=0 and j+1>=0 and j+1<=self.block_width-1:
						edges_index[0].append(back_node_num-now_node_num+i*self.block_width+j)
						edges_index[1].append(back_node_num-now_node_num+i*self.block_width+j+1)
						edges_attr.append(np.array([1,1]).tolist())
						
						edges_index[1].append(back_node_num-now_node_num+i*self.block_width+j)
						edges_index[0].append(back_node_num-now_node_num+i*self.block_width+j+1)
						edges_attr.append(np.array([1,1]).tolist())
					
			labels.append(label)
			u.append(u_temp)
		return tf.cast(self.List2Tensor(nodes),dtype=tf.float32),tf.cast(self.List2Tensor(edges_index),dtype=tf.int32),tf.cast(self.List2Tensor(edges_attr),dtype=tf.float32),tf.cast(self.List2Tensor(u),dtype=tf.float32),tf.cast(self.List2Tensor(batch),dtype=tf.int32),tf.one_hot(labels,36)

"""

batch_size = 3
img_height = 100
img_width = 56
stride = 8

code = Code("../../pytorch_verification_code/dataset/train",batch_size,stride,img_width,img_height)

nodes,edges_index,edges_attr,u,batch,labels = code.next_batch(7)


print(np.array(nodes).shape)
print(np.array(edges_index).shape)
print(np.array(edges_attr).shape)
print(np.array(u).shape)
print(np.array(batch).shape)
print(np.array(labels).shape)
print(labels)
"""
