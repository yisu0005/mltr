from .click_log import *

def read_reldocs(querypath, scorepath):
    with open(querypath, 'r') as f, open(scorepath, 'r') as g:
        lines = f.readlines()
        scores = g.readlines()
        number = len(lines)
        ids = set()
        queries = []
        for i in range(number):
            if int(lines[i][0]) == 0:
                info = lines[i].split(" ", 2)
                newid = int(info[1][4:])
                newdoc = Document(int(info[0]), info[2])
                newdoc.add_score(float(scores[i].rstrip()))
            elif int(lines[i][0]) == 1:
                info = lines[i].split(" ", 3)
                newid = int(info[1][4:])
                newdoc = Document(int(info[0]), info[3])
                newdoc.add_score(float(scores[i].rstrip()))
            if newid not in ids:
                ids.add(newid)
                newquery = Query(newid)
                newquery.add_doc(newdoc)
                queries.append(newquery)
            else:
                queries[-1].add_doc(newdoc)
    return queries


def true_avg_rank(queries):
    result = 0
    num = 0
    for query in queries:
        num += 1
        for doc in query.docs:
            if doc.rel == 1:
                result += doc.rank
    return result/num

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query_path', help='query path')
    parser.add_argument('score_path', help='prediction path')
    parser.add_argument('-c', '--c', default=0, help='hyperparameter value c')
    FLAGS, unparsed = parser.parse_known_args()

    queries = read_reldocs(FLAGS.query_path, FLAGS.score_path)

    for i in range(len(queries)):
        query_rank(queries[i])

    true_result = true_avg_rank(queries)
    print("{}:{}".format(FLAGS.c, true_result))


if __name__ == '__main__':
    main()
