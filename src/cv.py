from .click_log import *


def read_clickdocs(querypath, scorepath):
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
                newdoc.add_click(0)
            elif int(lines[i][0]) == 1:
                info = lines[i].split(" ", 3)
                newid = int(info[1][4:])
                newdoc = Document(int(info[0]), info[3])
                newdoc.add_score(float(scores[i].rstrip()))
                newdoc.add_click(1)
                newdoc.add_cost(float(info[2][5:]))
            if newid not in ids:
                ids.add(newid)
                newquery = Query(newid)
                newquery.add_doc(newdoc)
                queries.append(newquery)
            else:
                queries[-1].add_doc(newdoc)
    return queries

def ips_avg_rank(queries):
    result = 0
    num = 0
    for query in queries:
        for doc in query.docs:
            if doc.click == 1:
                result += doc.cost * doc.rank
                num += 1
    return result/num

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query_path', help='query path')
    parser.add_argument('score_path', help='prediction path')
    parser.add_argument('-c', '--c', help='hyperparameter value c')
    FLAGS, unparsed = parser.parse_known_args()

    queries = read_clickdocs(FLAGS.query_path, FLAGS.score_path)

    for i in range(len(queries)):
        query_rank(queries[i])

    ips_result = ips_avg_rank(queries)
    print("{}:{}".format(FLAGS.c, ips_result))


if __name__ == '__main__':
    main()