
import sonnet as snt
import tensorflow as tf


class Mish(snt.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, x):
       
        return x * tf.math.tanh(tf.math.softplus(x))

        
class EdgeModel(snt.Module):
	def __init__(self,OUTPUT_EDGE_SIZE):
		super(EdgeModel, self).__init__()
		self.OUTPUT_EDGE_SIZE = OUTPUT_EDGE_SIZE
		
		self.edge_mlp = snt.Sequential([
			snt.Linear(1024),
			Mish(),
			snt.Linear(self.OUTPUT_EDGE_SIZE)
		])
		
		#nodes.shape(1)*2+edge_attr.shape(1)+u.shape(1)
	def __call__(self, src, dest, edge_attr, u, batch):
		# source, target: [E, F_x], where E is the number of edges.
		# edge_attr: [E, F_e]
		# u: [B, F_u], where B is the number of graphs.
		# batch: [E] with max entry B - 1.
		out = tf.concat([src, dest, edge_attr, tf.gather(u, batch.numpy())], 1)
		return self.edge_mlp(out)

class NodeModel(snt.Module):
	def __init__(self,OUTPUT_NODE_SIZE):
		super(NodeModel, self).__init__()
		self.OUTPUT_NODE_SIZE = OUTPUT_NODE_SIZE
		
		self.node_mlp_1 = snt.Sequential([
			snt.Linear(1024),
			Mish(),
			snt.Linear(self.OUTPUT_NODE_SIZE)
		])
		
		self.node_mlp_2 = snt.Sequential([
			snt.Linear(1024),
			Mish(),
			snt.Linear(self.OUTPUT_NODE_SIZE)
		])
		
		#nodes.shape(1)+edge_attr.shape(1)
		#nodes.shape(1)*2+u.shape(1)
	def __call__(self, x, edge_index, edge_attr, u, batch):
		# x: [N, F_x], where N is the number of nodes.
		# edge_index: [2, E] with max entry N - 1.
		# edge_attr: [E, F_e]
		# u: [B, F_u]
		# batch: [N] with max entry B - 1.
		row, col = edge_index
		out = tf.concat([tf.gather(x, row.numpy()), edge_attr], 1)
		out = self.node_mlp_1(out)
		out = tf.compat.v2.math.unsorted_segment_mean(out, col, num_segments=x.shape[0])
		out = tf.concat([x, out, tf.squeeze(tf.gather(u, batch.numpy()))], 1)
		return self.node_mlp_2(out)

class GlobalModel(snt.Module):
	def __init__(self,OUTPUT_GLOBAL_SIZE):
		super(GlobalModel, self).__init__()
		self.OUTPUT_GLOBAL_SIZE = OUTPUT_GLOBAL_SIZE
		
		self.global_mlp = snt.Sequential([
			snt.Linear(1024),
			Mish(),
			snt.Linear(self.OUTPUT_GLOBAL_SIZE)
		])
		#u.shape(1)+nodes.shape(1)
	def __call__(self, x, edge_index, edge_attr, u, batch):
		# x: [N, F_x], where N is the number of nodes.
		# edge_index: [2, E] with max entry N - 1.
		# edge_attr: [E, F_e]
		# u: [B, F_u]
		# batch: [N] with max entry B - 1.
		out = tf.concat([u, tf.compat.v2.math.unsorted_segment_mean(x, tf.squeeze(batch), num_segments=u.shape[0])], 1)
		return self.global_mlp(out)
		
class GraphNetwork(snt.Module):
    
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(GraphNetwork, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

    def __call__(self, x, edge_index, edge_attr=None, u=None, batch=None):
        
        row, col = edge_index
        #print(tf.gather(batch, row.numpy()))
        if self.edge_model is not None:
            edge_attr = self.edge_model(tf.gather(x, row.numpy()), tf.gather(x, col.numpy()), edge_attr, u,
                                        batch if batch is None else tf.squeeze(tf.gather(batch, row.numpy())))

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u


class Captcha(snt.Module):
	def __init__(self):
		super(Captcha,self).__init__()
		self.GN_1 = GraphNetwork(EdgeModel(32), NodeModel(96), GlobalModel(1024))
		self.GN_2 = GraphNetwork(EdgeModel(16), NodeModel(48), GlobalModel(512))
		self.GN_3 = GraphNetwork(EdgeModel(8), NodeModel(24), GlobalModel(256))
		self.GN_4 = GraphNetwork(EdgeModel(4), NodeModel(12), GlobalModel(64))
		self.GN_5 = GraphNetwork(EdgeModel(2), NodeModel(6), GlobalModel(36))
		
	def __call__(self, x, edge_index, edge_attr, u, batch):

		x_, edge_attr_, u_ = self.GN_1(x, edge_index, edge_attr, u, batch)
		x_, edge_attr_, u_ = self.GN_2(x_, edge_index, edge_attr_, u_, batch)
		x_, edge_attr_, u_ = self.GN_3(x_, edge_index, edge_attr_, u_, batch)
		x_, edge_attr_, u_ = self.GN_4(x_, edge_index, edge_attr_, u_, batch)
		x_, edge_attr_, u_ = self.GN_5(x_, edge_index, edge_attr_, u_, batch)
		
		return x_, edge_attr_, u_
		
