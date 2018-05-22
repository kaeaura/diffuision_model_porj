__author__ = " Jing-Kai Lou and Fu-min Wang "
__email__ = "kaeaura@gmail.com"

import os
import sys
import math
import random
import numpy as np
import networkx as nx

def norm(l):
	"""
		unitify the vector
	"""
	try:
		return(map(lambda x: x/float(sum(l)) if sum(l) else 0, l))
	except TypeError:
		print('unexpected datatype')

def get_norm_matrix(mtx, axis = 1):
	"""
		Make each row in the matrix sum up 1
	"""
	try:
		mtx = np.array(mtx) if axis else np.array(mtx).T
		norm_mtx = np.matrix(map(norm, mtx.tolist()))
		return(norm_mtx if axis else norm_mtx.T)
	except TypeError:
		print ('type should be matrix')

def linearthreshold(graph, seeds):
	"""
		get the casceded status based on linear threshold model, a dictionary is returned
	"""
	m = LinearThreshold(graph, seeds)
	m.propagate()
	return(m.show_status())

class LinearThreshold:
	"""
		Jon Kleinberg, E. T. (2003). Maximizing the Spread of Influence through a Social Network, 1-10.
	"""
	def __init__(self, graph, seeds, weight = 0.01, global_threshold = 0.01):
		if graph.is_directed is False:
			graph = graph.to_directed()
		self.graph = graph
		self.seeds = self.set_seeds(seeds)
		self.set_threshold(global_threshold)
		self.set_linkweight(weight)
		nx.set_node_attributes(self.graph, 'pressure', dict(zip(self.graph.nodes(), [0] * self.graph.order())))

	def set_seeds(self, seeds):
		in_seeds = filter(lambda x: x in self.graph.nodes(), seeds)
		if in_seeds != seeds:
			print 'warning! improper seed set'
		self.status = dict(zip(self.graph.nodes(), map(lambda x: 1 if x in in_seeds else 0, self.graph.nodes())))
		nx.set_node_attributes(self.graph, 'status', self.status)

	def set_threshold(self, threshold):
		if type(threshold) is float:
			self.threshold = dict(zip(self.graph.nodes(), [threshold] * self.graph.order()))
		elif type(threshold) is list:
			self.threshold = dict(zip(self.graph.nodes(), threshold))
		else:
			print ('arg: threshold setting error')
			sys.exit(2)
		nx.set_node_attributes(self.graph, 'thr', self.threshold)

	def set_linkweight(self, weight):
		if type(weight) is dict:
			try:
				nx.set_edge_attributes(self.graph, 'weight', weight)
			except:
				print ('arg: weight seeting error')
		elif type(weight) is float or type(weight) is int:
			nx.set_edge_attributes(self.graph, 'weight', dict(zip(self.graph.edges(), [weight] * self.graph.size())))

	def collapse_node(self, node):
		collapsed_neis = []
		try:
			for nei in self.graph.successors(node):
				if self.graph.node[node]['status'] == 1:
					self.graph.node[nei]['pressure'] += self.graph[node][nei]['weight']
				if self.graph.node[nei]['pressure'] >= self.graph.node[nei]['thr'] and self.graph.node[nei]['status'] == 0:
					self.graph[node][nei]['cri'] = 1
					self.graph.node[nei]['status'] = 1
					collapsed_neis.append(nei)
			return(collapsed_neis)
		except KeyError:
			raise

	def propagate(self):
		propagate_queue_items = filter(lambda x: x[1] == 1, nx.get_node_attributes(self.graph, 'status').items())
		propagate_queue = map(lambda x: x[0], propagate_queue_items)
		finish_queue = []
		while(len(propagate_queue)):
			propagate_node = propagate_queue.pop()
			collpased_nodes = self.collapse_node(propagate_node)
			finish_queue.append(propagate_node)
			if len(collpased_nodes) > 0:
				propagate_queue.extend(collpased_nodes)

	def show_status(self):
		return(nx.get_node_attributes(self.graph, 'status'))

def diffusionrank(graph, heat, alpha = 0.85, gamma = 1, N = 100):
	"""
		get the heat values based on heat diffusion model, a dictionary is returned
	"""
	m = DiffusionRank(graph, heat)
	return(m.get_d_erH(alpha, gamma, N) * m.heat)

class DiffusionRank:
	"""
		Yang, H., King, I., & Lyu, M. R. (2007). DiffusionRank: a possible penicillin for web spamming. In SIGIR '07: Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval.
	"""
	def __init__(self, graph, heat):
		self.graph = graph
		self.heat = np.matrix(heat).reshape(graph.order(), 1)
		self.I = np.diag([1] * graph.order())
		self.H = self.get_heat_matrix()

	def get_heat_matrix(self):
		adj_mtx = nx.convert_matrix.to_numpy_matrix(self.graph)
		heat_mtx = -1 * np.eye(self.graph.order()) + get_norm_matrix(adj_mtx, axis = 0)
		return(heat_mtx)

	def get_d_erH(self, alpha = 1, gamma = 1, N = 100):
		p_mtx = nx.google_matrix(self.graph, alpha = alpha)
		r_mtx = -1 * np.eye(self.graph.order()) + p_mtx
		d_erH = (np.eye(self.graph.order()) + (float(gamma) / N ) * r_mtx) ** N
		return(d_erH)

def independentcascade(graph, seeds):
	"""
		get the casceded status based on independent cascade model, a list is returned
	"""
	m = IndependentCascade(graph, seeds)
	return(m.iter())

class IndependentCascade:
	def __init__(self, graph, seeds):
		self.graph = graph
		self.seeds = seeds
		self.prob_degree()

	def prob_degree(self):
		for n, nbrs in self.graph.adjacency_iter():
			for nbr, eattr in nbrs.items():
				eattr['prob'] = 1.0 / self.graph.out_degree(n)

	def try_touch(self, prob):
		return (True if random.random() < prob else False)

	def iter(self):
		for n in self.seeds:
			self.graph.node[n]['touched'] = True
			self.graph.node[n]['activep'] = True
		diffset = set(self.graph.nodes()) - set(self.seeds)
		for n in diffset:
			self.graph.node[n]['touched'] = False
			self.graph.node[n]['touched'] = False
		ans = subs = self.seeds
		while subs:
			temp_subs = []
			for n in subs:
				for nbr in self.graph.neighbors(n):
					if self.graph.node[nbr]['touched'] == False:
						self.graph.node[nbr]['activep'] = self.try_touch(self.graph[n][nbr]['prob'])
						if self.graph.node[nbr]['activep']:
							temp_subs.append(nbr)
						self.graph.node[nbr]['touched'] = True
			subs = temp_subs
			ans = ans+subs
		return (ans)
