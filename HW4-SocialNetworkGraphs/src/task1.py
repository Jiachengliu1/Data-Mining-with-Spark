from graphframes import GraphFrame
import os
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
import time


os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")


if __name__ == '__main__':
	start = time.time()
	filter_threshold = int(sys.argv[1])
	f_in = sys.argv[2]
	f_out = sys.argv[3]

	sc = SparkContext.getOrCreate()
	sc.setLogLevel('WARN')	
	sqlc = SQLContext(sc)

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
					vertices.add((l1[0],))
					vertices.add((l2[0],))
					edges.add(tuple((l1[0], l2[0])))

	# LPA
	vertices = sqlc.createDataFrame(list(vertices), ['id'])
	edges = sqlc.createDataFrame(list(edges), ['src', 'dst'])
	g = GraphFrame(vertices, edges)
	result = g.labelPropagation(maxIter=5)
	#print(result.collect())

	communities = result.rdd.map(lambda x: (x[1], x[0])) \
	.groupByKey().map(lambda x: list(sorted(x[1]))) \
	.sortBy(lambda x: (len(x), x)).collect()

	with open(f_out, 'w+') as fout:
		for community in communities:
			fout.write(str(community).strip('[]') + '\n')

	end = time.time()
	print('Duration: {}'.format(end - start))

