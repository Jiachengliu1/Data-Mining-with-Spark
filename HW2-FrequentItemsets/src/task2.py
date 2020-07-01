from itertools import combinations
import math
from operator import add
from pyspark import SparkContext
import sys
import time


def get_C1(dataset):

    C1 = dict()
    for li in dataset:
        for element in li:
            if element not in C1:
                C1[element] = 1
            else:
                C1[element] += 1

    return C1


def Lk_to_Ck(dataset, Lk, k):

    Ck = dict()
    for li in dataset:
        li = sorted(set(li) & set(Lk))
        #print('li:', li)
        for item in combinations(li, k):
            item = tuple(item)
            if item not in Ck:
                Ck[item] = 1
            else:
                Ck[item] += 1

    return Ck


def Ck_to_Lk(Ck, support):

    Lk = list()
    for key, val in Ck.items():
        if val >= support:
            Lk.append(key)

    return sorted(Lk)


def find_candidates(dataset, support, whole_size):

    frequent_itemsets = []

    # initialize partition support threshold
    partition = list(dataset)
    p = len(partition) / whole_size
    ps =  math.ceil(p * support)

    # get all singletons
    C1 = get_C1(partition)
    #print('C1:', C1)
    L1 = Ck_to_Lk(C1, ps)
    # str to tuple
    frequent_itemsets.append([(item,) for item in L1])

    # initialize combination length
    k = 2

    while True:
        #print('L1:',L1)
        Ck = Lk_to_Ck(partition, L1, k)
        #print('Ck:',Ck)
        Lk = Ck_to_Lk(Ck, ps)
        if Lk == []:
            break
        frequent_itemsets.append(Lk)
        # update L1
        L1 = set()
        for item in Lk:
            L1 = L1 | set(item)
        k += 1

    return frequent_itemsets


def find_frequent_itemsets(dataset, candidates):

    result_dict = dict()
    for li in dataset:
        for item in candidates:
            if set(item).issubset(li):
                if item not in result_dict:
                    result_dict[item] = 1
                else:
                    result_dict[item] += 1

    result_li = [(key, value) for key, value in result_dict.items()]

    return result_li


def format(data):

    result = ''
    length = 1
    for item in data:
        if len(item) == 1:
            result += str(item).replace(',', '') + ','
        elif len(item) == length:
            result += str(item) + ','
        else:
            result += '\n\n'
            result += str(item) + ','
            length = len(item)

    result = result.replace(',\n\n', '\n\n')[:-1]

    return result


if __name__ == '__main__':
    start = time.time()
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    sc = SparkContext.getOrCreate()

    # Generate basket model
    rdd = sc.textFile(input_file_path)
    header = rdd.first()

    new_rdd = rdd.filter(lambda x: x != header) \
    .map(lambda x: (x.split(',')[0], x.split(',')[1])) \
    .groupByKey() \
    .map(lambda x: list(set(x[1]))) \
    .filter(lambda x: len(x) > filter_threshold)


    whole_size = new_rdd.count()
    #print(whole_size)

    # Phase 1
    candidates = new_rdd.mapPartitions(lambda partition: find_candidates(partition, support, whole_size)) \
    .flatMap(lambda x: x) \
    .distinct() \
    .sortBy(lambda x: (len(x), x)).collect()

    #print(candidates)

    # Phase 2
    frequent_itemsets = new_rdd.mapPartitions(lambda partition: find_frequent_itemsets(partition, candidates)) \
    .reduceByKey(add) \
    .filter(lambda x: x[1] >= support) \
    .map(lambda x: x[0]) \
    .sortBy(lambda x: (len(x), x)).collect()

    #print(format(frequent_itemsets))

    with open(output_file_path, 'w+') as fout:
        fout.write('Candidates:\n' + format(candidates) + '\n\n' + 'Frequent Itemsets:\n' + format(frequent_itemsets))

    end = time.time()
    print('Duration: {}'.format(end - start))

