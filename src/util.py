import argparse
import math
import os
import random
import timeit


class Query(object):
    def __init__(self, qid):
        self.qid = qid
        self.docs = []

    def add_doc(self, document):
        self.docs.append(document)

    def add_clickid(self, clickid):
        self.clickid = clickid

    def del_clicks(self):
        for doc in self.docs:
            doc.add_click(0)


class Document(object):
    def __init__(self, rel, feature):
        self.rel = rel
        self.feature = feature

    # Recommendation: get_score / set_score
    def add_score(self, score):
        self.score = score

    def add_rank(self, rank):
        self.rank = rank

    def add_click(self, click):
        self.click = click

    # the following two function are same; maybe set_cost?
    def add_cost(self, cost):
        self.cost = cost

    def change_cost(self, newcost):
        self.cost = newcost


def random_draw(rankerA_prec, rankerB_prec, overlap, index):
    indexList = list(range(index))
    random.shuffle(indexList)

    assert overlap <= min(rankerA_prec, rankerB_prec)

    rankerB_start = int(math.ceil((rankerA_prec - overlap) * index))
    rankerB_end = int(math.ceil((rankerA_prec - overlap + rankerB_prec) * index))
    rankerA_index = indexList[:int(math.ceil(rankerA_prec * index))]
    rankerB_index = indexList[rankerB_start:rankerB_end]
    rankerA_index.sort()
    rankerB_index.sort()

    return rankerA_index, rankerB_index


def read_doc(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        number = len(lines)
        ids = set()
        queries = []
        for i in range(number):
            info = lines[i].split(" ", 2)
            newid = int(info[1][4:])
            newdoc = Document(int(info[0]), info[2])
            if newid not in ids:
                ids.add(newid)
                newquery = Query(newid)
                newquery.add_doc(newdoc)
                queries.append(newquery)
            else:
                queries[-1].add_doc(newdoc)
    return queries


def save_doc(path, rankerA_query, rankerB_query):
    if not os.path.exists(path):
        os.makedirs(path)
    queryList = [rankerA_query, rankerB_query]
    queryname = ['rankerA_query', 'rankerB_query']

    # FIXME: features include '\n'...
    for i in range(2):
        with open(os.path.join(path, queryname[i] + '.txt'), 'w') as f:
            for query in queryList[i]:
                for doc in query.docs:
                    f.write(str(doc.rel) + " " + "qid:" + str(query.qid) + " " + str(doc.feature))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('-pr', '--ranker_prop', help='ranker training proportion [rankerA, rankerB, overlap]',
                        nargs='+', type=float)
    FLAGS, unparsed = parser.parse_known_args()

    start = timeit.default_timer()
    path = FLAGS.input_dir
    queries = read_doc(path)
    rankerA_index, rankerB_index = random_draw(FLAGS.ranker_prop[0], FLAGS.ranker_prop[1], FLAGS.ranker_prop[2],
                                               len(queries))
    rankerA_query = [queries[i] for i in rankerA_index]
    rankerB_query = [queries[i] for i in rankerB_index]
    save_doc(FLAGS.output_dir, rankerA_query, rankerB_query)

    end = timeit.default_timer()
    print('Finished preparing data for training rankers')
    print('Running time: {:.3f}s.'.format(end - start))


if __name__ == '__main__':
    main()
