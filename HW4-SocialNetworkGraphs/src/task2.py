from collections import defaultdict
from operator import add
from pyspark import SparkContext
import random
import sys
import time


def Girvan_Newman(root):

	# build structure: tree: {level 1: [user1, user2]}
	tree = dict()
	find_parent = defaultdict(set)
	find_child = dict()
	num_path = dict()

	# initialization
	tree[0] = root
	num_path[root] = 1
	used_nodes = {root}
	level_1_nodes = adjacent_nodes[root]
	find_child[root] = level_1_nodes
	level = 1

	while level_1_nodes != set():
		tree[level] = level_1_nodes
		used_nodes = used_nodes.union(level_1_nodes)
		new_nodes = set()
		for node in level_1_nodes:
			adj_nodes = adjacent_nodes[node]
			# find_child: {node, [list of child nodes]}
			child_nodes = adj_nodes - used_nodes
			find_child[node] = child_nodes
			# find_parent: {node, [list of parent nodes]}
			for key, vals in find_child.items():
				for val in vals:
					find_parent[val].add(key)
			# find number of shortest paths of each node
			parent_nodes = find_parent[node]
			if len(parent_nodes) > 0:
				num_path[node] = sum([num_path[parent_node] for parent_node in parent_nodes])
			else:
				num_path[node] = 1
			new_nodes = new_nodes.union(adj_nodes)
		level_nodes = new_nodes - used_nodes
		level_1_nodes = level_nodes
		level += 1

	# calculate betweenness for each edge
	parent_value = defaultdict(float)
	for node in vertices:
		if node != root:
			parent_value[node] = 1
	
	edge_value = dict()

	while level != 1:
		for node in tree[level - 1]:
			parent_nodes = find_parent[node]
			#weight = 1.0 / len(parent_nodes)
			for parent_node in parent_nodes:
				weight = num_path[parent_node] / num_path[node]
				edge_value[tuple(sorted((node, parent_node)))] = weight * parent_value[node]
				parent_value[parent_node] += edge_value[tuple(sorted((node, parent_node)))]
		level -= 1

	return [(key, val) for key, val in edge_value.items()]


def find_community(node, adjacent_nodes):

	used_nodes = set()
	community = set()
	count = 0
	adj_nodes = adjacent_nodes[node]
	while True:
		used_nodes = used_nodes.union(adj_nodes)
		count += 1
		new_nodes = set()
		for n in adj_nodes:
			new_adj_nodes = adjacent_nodes[n]
			new_nodes = new_nodes.union(new_adj_nodes)
		new_used_nodes = used_nodes.union(new_nodes)
		# check if current community is finished
		if len(used_nodes) == len(new_used_nodes):
			break
		adj_nodes = new_nodes - used_nodes

	community = used_nodes
	if community == set():
		community = {node}

	return community


def find_communities(node, vertices, adjacent_nodes):

	communities = []
	used_nodes = find_community(node, adjacent_nodes)
	unused_nodes = vertices - used_nodes
	communities.append(used_nodes)
	while True:
		new_used_nodes = find_community(random.sample(unused_nodes, 1)[0], adjacent_nodes)
		communities.append(new_used_nodes)
		used_nodes = used_nodes.union(new_used_nodes)
		unused_nodes = vertices - used_nodes
		# check if all communities have been found
		if len(unused_nodes) == 0:
			break

	return communities


def calculate_modularity(communities, m):

	modularity = 0
	for community in communities:
		partition_modularity = 0
		for i in community:
			for j in community:
				partition_modularity += A[(i, j)] - degree[i] * degree[j] / (2 * m)
		modularity += partition_modularity
	modularity = modularity / (2 * m)

	return modularity


if __name__ == '__main__':
	start = time.time()
	filter_threshold = int(sys.argv[1])
	f_in = sys.argv[2]
	f_out_betweenness = sys.argv[3]
	f_out_community = sys.argv[4]

	sc = SparkContext.getOrCreate()
	sc.setLogLevel('WARN')

	# generate structure: [user_id, [business_id1, business_id2]]
	rdd = sc.textFile(f_in)
	header = rdd.first()
	user_business = rdd.filter(lambda x: x != header) \
    .map(lambda x: (x.split(',')[0], x.split(',')[1])) \
    .groupByKey().map(lambda x: (x[0], list(set(x[1])))).collect()

	# generate vertices and edges such that two users review the same business >= 7
	vertices = set()
	edges = set()
	for l1 in user_business:
		for l2 in user_business:
			if l1[0] != l2[0]:
				if len(set(l1[1]) & set(l2[1])) >= filter_threshold:
					vertices.add(l1[0])
					vertices.add(l2[0])
					edges.add(tuple((l1[0], l2[0])))

	# generate structure {node: [list of adjacent nodes]}
	adjacent_nodes = defaultdict(set)
	for pair in edges:
		adjacent_nodes[pair[0]].add(pair[1])

	betweenness = sc.parallelize(vertices).map(lambda node: Girvan_Newman(node)) \
	.flatMap(lambda x: [pair for pair in x]) \
	.reduceByKey(add) \
	.map(lambda x: (x[0], x[1] / 2)) \
	.sortBy(lambda x: (-x[1], x[0])).collect()

	with open(f_out_betweenness, 'w+') as fout:
		for pair in betweenness:
			fout.write(str(pair)[1:-1] + '\n')

	# degree mapping
	degree = dict()
	for key, val in adjacent_nodes.items():
		degree[key] = len(val)
	
	# adjacent matrix mapping
	A = dict()
	for node1 in vertices:
		for node2 in vertices:
			if (node1, node2) in edges:
				A[(node1, node2)] = 1
			else:
				A[(node1, node2)] = 0

	# edge number of the original graph m
	m = len(edges) / 2

	left_edges = m
	max_modularity = -1

	while True:
		highest_betweenness = betweenness[0][1]
		for pair in betweenness:
			# remove this pair
			if pair[1] == highest_betweenness:
				# update adjacent_nodes in order to find communities
				adjacent_nodes[pair[0][0]].remove(pair[0][1])
				adjacent_nodes[pair[0][1]].remove(pair[0][0])
				left_edges -= 1

		# find communities
		temp_communities = find_communities(random.sample(vertices, 1)[0], vertices, adjacent_nodes)

		# calculate current modularity
		cur_modularity = calculate_modularity(temp_communities, m)
		
		# update max_modularity and communities
		if cur_modularity > max_modularity:
			max_modularity = cur_modularity
			communities = temp_communities

		# repeat until no edges are left
		if left_edges == 0:
			break

		# update betwennness
		betweenness = sc.parallelize(vertices).map(lambda node: Girvan_Newman(node)) \
		.flatMap(lambda x: [pair for pair in x]) \
		.reduceByKey(add) \
		.map(lambda x: (x[0], x[1] / 2)) \
		.sortBy(lambda x: (-x[1], x[0])).collect()

	sorted_communities = sc.parallelize(communities) \
	.map(lambda x: sorted(x)) \
	.sortBy(lambda x: (len(x), x)).collect()

	with open(f_out_community, 'w+') as fout:
		for community in sorted_communities:
			fout.write(str(community)[1:-1] + '\n')

	end = time.time()
	print('Duration: {}'.format(end - start))

