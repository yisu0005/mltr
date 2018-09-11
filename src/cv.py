from .click_log import *

'''
calculate the IPS estimator of ave rel rank
'''

def read_clickdocs(querypath, scorepath):
    with open(querypath, 'r') as f, open(scorepath, 'r') as g:
        ids = set()
        queries = []
        for line, score in zip(f, g):
            if line[0] == '0':
                info = line.rstrip().split(' ', 2)
                newid = int(info[1][4:])
                newdoc = Document(int(info[0]), info[2])
                newdoc.add_score(float(score.rstrip()))
            elif line[0] == '1':
                info = line.split(" ", 3)
                newid = int(info[1][4:])
                newdoc = Document(int(info[0]), info[3])
                newdoc.add_score(float(score.rstrip()))
                newdoc.set_click(1)
                newdoc.set_cost(float(info[2][5:]))
            if newid not in ids:
                ids.add(newid)
                newquery = Query(newid)
                newquery.add_doc(newdoc)
                queries.append(newquery)
            else:
                queries[-1].add_doc(newdoc)
    return queries

def ips_avg_rank(queries, sweep):
    result = 0
    num = 0
    for query in queries:
        num += 1
        for doc in query.docs:
            if doc.click == 1:
                result += doc.cost * doc.rank
    return result/(6983*sweep*2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query_path', help='query path')
    parser.add_argument('score_path', help='prediction path')
    parser.add_argument('-s', help='number of sweeps', type=int)
    parser.add_argument('-c', '--c', default=0, help='hyperparameter value c')
    FLAGS, unparsed = parser.parse_known_args()

    queries = read_clickdocs(FLAGS.query_path, FLAGS.score_path)

    for i in range(len(queries)):
        query_rank(queries[i])

    ips_result = ips_avg_rank(queries, FLAGS.s)
    print("{}:{}".format(FLAGS.c, ips_result))


if __name__ == '__main__':
    main()
