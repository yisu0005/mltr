import copy
import numpy as np
from .util import *

def query_rank(query):
    scores = []
    for i in range(len(query.docs)):
        scores.append(query.docs[i].score)
    ranks = np.argsort(np.argsort(np.array(scores) * (-1)))

    for i in range(len(query.docs)):
        query.docs[i].add_rank(ranks[i])


def draw_prob(prob):
    return random.random() < prob


def get_propensity(rank, eta):
    return np.power(1.0/(rank + 1), eta)


def get_random_idx(lengthA, query0_num):
    assert query0_num <= lengthA
    query0_idx = random.sample(range(lengthA), k=query0_num)
    return sorted(query0_idx)


def make_click(queries, eta, eps_plus, eps_minus, sweep):

    for query in queries:
        query_rank(query)
        for doc in query.docs:
            propensity = get_propensity(doc.rank, eta)
            doc.set_cost(1.0 / propensity)

    for i in range(sweep):
        for query in queries:
            for doc in query.docs:
                obs = draw_prob(1.0/doc.cost)
                if obs and doc.rel == 1:
                    if draw_prob(eps_plus):
                        doc.add_click()
                elif obs and doc.rel == 0:
                    if draw_prob(eps_minus):
                        doc.add_click()


def single_doc(queries):
    queries_cp = copy.deepcopy(queries)
    tempqueries = []
    for i in range(len(queries_cp)):
        queries_cp[i].del_clicks()
        tempqueries.append(queries_cp[i])

    newqueries = []
    for i in range(len(queries)):
        for j in range(len(queries[i].docs)):
            cnt = queries[i].docs[j].click
            if cnt > 0:
                for t in range(cnt):
                    q = copy.deepcopy(tempqueries[i])
                    q.docs[j].set_click(1)
                    newqueries.append(q)

    clickid = 1
    for query in newqueries:
        query.add_clickid(clickid)
        clickid += 1
    return newqueries


def get_clickNo(queriesA, queriesB):
    n1, n2 = 0, 0
    for query in queriesA:
        n1 += np.sum([doc.click for doc in query.docs])
    for query in queriesB:
        n2 += np.sum([doc.click for doc in query.docs])
    return n1, n2


def balance_prop(queriesA, queriesB, n1, n2):
    assert len(queriesA) == len(queriesB)
    queriesAnew = copy.deepcopy(queriesA)
    queriesBnew = copy.deepcopy(queriesB)
    for qa, qb in zip(queriesAnew, queriesBnew):
        docs_a, docs_b = qa.docs, qb.docs
        assert len(docs_a) == len(docs_b)
        for doc_a, doc_b in zip(docs_a, docs_b):
            propA, propB = 1.0/doc_a.cost, 1.0/doc_b.cost
            prop = (n1 * propA + n2 * propB) / (n1 + n2)
            cost = 1.0 / prop
            doc_a.set_cost(cost)
            doc_b.set_cost(cost)
    with open('clickrank.txt', 'w') as fout:
        for qa, qb in zip(queriesAnew, queriesBnew):
            docs_a, docs_b = qa.docs, qb.docs
            assert len(docs_a) == len(docs_b)
            for doc_a, doc_b in zip(docs_a, docs_b):
                if doc_a.click == 1 or doc_b.click == 1:
                    fout.write("{} and {}\n".format(doc_a.rank, doc_b.rank))

    return queriesAnew, queriesBnew


def clip_prop(queriesA, queriesB, c):
    assert len(queriesA) == len(queriesB)
    queriesAnew = copy.deepcopy(queriesA)
    queriesBnew = copy.deepcopy(queriesB)
    for qa, qb in zip(queriesAnew, queriesBnew):
        docs_a, docs_b = qa.docs, qb.docs
        assert len(docs_a) == len(docs_b)
        for doc_a, doc_b in zip(docs_a, docs_b):
            propA, propB = max(1.0/doc_a.cost, c), max(1.0/doc_b.cost, c)
            costA, costB = 1.0/propA, 1.0/propB
            doc_a.set_cost(costA)
            doc_b.set_cost(costB)
    return queriesAnew, queriesBnew


def read_docs(querypath, scorepath):
    with open(querypath, 'r') as f, open(scorepath, 'r') as g:
        lines = f.readlines()
        scores = g.readlines()
        number = len(lines)
        ids = set()
        queries = []
        for i in range(number):
            info = lines[i].split(" ", 2)
            newid = int(info[1][4:])
            newdoc = Document(int(info[0]), info[2])
            newdoc.add_score(float(scores[i].rstrip()))
            if newid not in ids:
                ids.add(newid)
                newquery = Query(newid)
                newquery.add_doc(newdoc)
                queries.append(newquery)
            else:
                queries[-1].add_doc(newdoc)
    return queries


def save_propfile(path, name, A_query, B_query):
    if path and not os.path.exists(path):
        os.makedirs(path)
    queryList = [A_query, B_query]
    queryA_max = A_query[-1].clickid

    with open(os.path.join(path, name), 'w') as f:
        for i in range(2):
            for query in queryList[i]:
                for doc in query.docs:
                    if doc.click:
                        f.write(str(doc.click) + " " + "qid:" + str(query.clickid + queryA_max * i) + " " + "cost:" + str(doc.cost) + " " + str(
                                doc.feature))
                    else:
                        f.write(str(doc.click) + " " + "qid:" + str(query.clickid + queryA_max * i) + " " + str(doc.feature))


def delete_noclick(queries):
    cleared_queries = []
    for query in queries:
        allclick = False
        for doc in query.docs:
            if doc.click == 1:
                allclick = True
                break
        if allclick:
            cleared_queries.append(query)
    return cleared_queries


def get_prop_diagnostic(subquery0, subquery1):
    avg_prop = 0
    num = 0
    for queries in [subquery0, subquery1]:
        for query in queries:
            for doc in query.docs:
                if doc.click == 1:
                    avg_prop += 1.0/doc.cost
                    num += 1
    return avg_prop/num


def build_queryB(queriesA, change_option):
    queriesB = copy.deepcopy(queriesA)
    if change_option == 'lastone':
        for i in range(len(queriesA)):
            scores = []
            for j in range(len(queriesA[i].docs)):
                scores.append(queriesA[i].docs[j].score)
            max_idx = np.argmax(scores)
            min_idx = np.argmin(scores)
            max_val, min_val = queriesB[i].docs[max_idx].score, queriesB[i].docs[min_idx].score
            queriesB[i].docs[max_idx].add_score(min_val)
            queriesB[i].docs[min_idx].add_score(max_val)
    elif change_option == 'all':
        for i in range(len(queriesA)):
            for j in range(len(queriesA[i].docs)):
                temp_score = queriesA[i].docs[j].score
                queriesB[i].docs[j].add_score(-1 * temp_score)
    elif change_option == 'portion':
        PORTION = 0.8
        for q in queriesB:
            new_docs = sorted(q.docs, key=lambda d: d.score, reverse=True)
            num = int(len(q.docs) * (1 - PORTION))
            r_docs = new_docs[num:]
            rv_docs = list(reversed(r_docs))
            for i in range(len(r_docs)):
                r_docs[i].add_score(rv_docs[i].score)
            q.docs = new_docs[:num] + r_docs
    return queriesB


def reverse_scores(scores):
    in_index = np.argsort(np.argsort(-1 * np.array(scores)))
    new_score = sorted(scores)
    rev_scores = [new_score[i] for i in in_index.tolist()]
    return rev_scores


def build_rel_queryB(queriesA, change_option):
    queriesB = copy.deepcopy(queriesA)
    rel_indices_ls = []
    rel_scores_ls = []
    for i in range(len(queriesA)):
        rel_indices = []
        rel_scores = []
        for j in range(len(queriesA[i].docs)):
            if queriesA[i].docs[j].rel == 1:
                rel_indices.append(j)
                rel_scores.append(queriesA[i].docs[j].score)
        rel_indices_ls.append(rel_indices)
        rel_scores_ls.append(rel_scores)

    if change_option == 'lastone':
        for i in range(len(queriesA)):
            rel_indices = rel_indices_ls[i]
            rel_scores = rel_scores_ls[i]
            if rel_scores:
                max_idx = np.argmax(rel_scores)
                min_idx = np.argmin(rel_scores)
                max_val, min_val = max(rel_scores), min(rel_scores)
                queriesB[i].docs[rel_indices[max_idx]].add_score(min_val)
                queriesB[i].docs[rel_indices[min_idx]].add_score(max_val)
    elif change_option == 'all':
        for i in range(len(queriesA)):
            rel_indices = rel_indices_ls[i]
            rel_scores = rel_scores_ls[i]
            if rel_scores:
                rev_scores = reverse_scores(rel_scores)
                for k in range(len(rel_indices)):
                    queriesB[i].docs[rel_indices[k]].add_score(rev_scores[k])
    elif change_option == 'portion':
        PORTION = 0.8
        for i in range(len(queriesA)):
            rel_indices = rel_indices_ls[i]
            rel_scores = rel_scores_ls[i]
            if rel_scores:
                order = np.argsort(rel_scores)
                num = int(len(order) * (1 - PORTION))
                del_index = order[num:]
                new_rel_indices = [rel_indices[i] for i in \
                                range(len(rel_indices)) if i not in del_index]
                new_scores = [rel_scores[i] for i in \
                                range(len(rel_indices)) if i not in del_index]
                rev_new_scores = reverse_scores(new_scores)
                for k in range(len(new_rel_indices)):
                    queriesB[i].docs[new_rel_indices[k]].add_score(rev_new_scores[k])
    return queriesB

def click_diagonistic(queryA, queryB):
    for query in queryA:
        print("total len: {}".format(len(query.docs)))
        for doc in query.docs:
            # if doc.rel == 1:
            #     print(doc.rank)
            if doc.click == 1:
                print("rank: {}, rel: {} and cost {}.".format(doc.rank, doc.rel, doc.cost))
    for query in queryB:
        print("total len: {}".format(len(query.docs)))
        for doc in query.docs:
            # if doc.rel == 1:
            #     print(doc.rank)
            if doc.click == 1:
                print("rank: {}, rel: {} and cost {}.".format(doc.rank, doc.rel, doc.cost))







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query_path', help='query path')
    parser.add_argument('ranker0_path', help='ranker0 prediction path')
    parser.add_argument('ranker1_path', help='ranker1 prediction path')
    parser.add_argument('ranker1_rel', help='ranker1 rel change [0,1]', type=int)
    parser.add_argument('output_dict', help='output_directory')
    parser.add_argument('name_prop', help='clicked propensity file')
    parser.add_argument('name_bal', help='balanced propensity file')
    parser.add_argument('name_clip', help='clipped propensity file')
    parser.add_argument('name_clipbal', help='clipped balanced propensity file')
    parser.add_argument('-e', '--eta', help='propensity stochasity', type=float)
    parser.add_argument('-p', '--eps_plus', help='positive click noise', type=float)
    parser.add_argument('-m', '--eps_minus', help='negative click noise', type=float)
    parser.add_argument('-a', '--query0', help='number of clicks from first query', type=int)
    parser.add_argument('-b', '--query1', help='number of clicks from second query', type=int)
    parser.add_argument('-s', '--sweep', help='number of sweeps', type=int)
    FLAGS, unparsed = parser.parse_known_args()
    print('Building clicked logs. ')
    start = timeit.default_timer()

    queriesA = read_docs(FLAGS.query_path, FLAGS.ranker0_path)
    make_click(queriesA, FLAGS.eta, FLAGS.eps_plus, FLAGS.eps_minus,FLAGS.sweep)
    delqueriesA = delete_noclick(queriesA)
    newqueryA = single_doc(delqueriesA)
    idxA = get_random_idx(len(newqueryA), FLAGS.query0)


    # for rankerB_way in ['lastone', 'all', 'portion']:
    for rankerB_way in ['lastone']:
        if FLAGS.ranker1_rel == 1:
            print("build rel ranker B")
            queriesB = build_rel_queryB(queriesA, rankerB_way)
        elif FLAGS.ranker1_rel == 0:
            print("build switch all ranker B")
            queriesB = build_queryB(queriesA, rankerB_way)
        else:
            print("build ranker B with percentage of training data")
            queriesB = read_docs(FLAGS.query_path, FLAGS.ranker1_path)

        make_click(queriesB, FLAGS.eta, FLAGS.eps_plus, FLAGS.eps_minus, FLAGS.sweep)
        delqueriesB = delete_noclick(queriesB)
        newqueryB = single_doc(delqueriesB)
        idxB = get_random_idx(len(newqueryB), FLAGS.query1)

        subqueryA = [newqueryA[i] for i in idxA]
        subqueryB = [newqueryB[i] for i in idxB]
        # with open('clickresultnaive.txt', 'w') as fout:
        #     for query in subqueryA:
        #         for doc in query.docs:
        #             if doc.click == 1:
        #                 fout.write("{}\n".format(1.0/doc.cost))

        save_propfile(os.path.join(FLAGS.output_dict, rankerB_way), FLAGS.name_prop, subqueryA, subqueryB)
        prop_before = get_prop_diagnostic(subqueryA, subqueryB)


        n1, n2 = get_clickNo(subqueryA, subqueryB)
        bal_queriesA, bal_queriesB = balance_prop(queriesA, queriesB, n1, n2)
        del_bal_queriesA = delete_noclick(bal_queriesA)
        del_bal_queriesB = delete_noclick(bal_queriesB)
        new_bal_queriesA = single_doc(del_bal_queriesA)
        new_bal_queriesB = single_doc(del_bal_queriesB)
        bal_subqueryA = [new_bal_queriesA[i] for i in idxA]
        bal_subqueryB = [new_bal_queriesB[i] for i in idxB]
        # with open('clickresultbal.txt', 'w') as fout:
        #     for query in bal_subqueryA:
        #         for doc in query.docs:
        #             if doc.click == 1:
        #                 fout.write("{}\n".format(1.0/doc.cost))
        prop_after = get_prop_diagnostic(bal_subqueryA, bal_subqueryB)
        save_propfile(os.path.join(FLAGS.output_dict, rankerB_way), FLAGS.name_bal, bal_subqueryA, bal_subqueryB)

        c = 0.1
        clip_queriesA, clip_queriesB = clip_prop(queriesA, queriesB, c)
        del_clip_queriesA = delete_noclick(clip_queriesA)
        del_clip_queriesB = delete_noclick(clip_queriesB)
        new_clip_queriesA = single_doc(del_clip_queriesA)
        new_clip_queriesB = single_doc(del_clip_queriesB)
        clip_subqueryA = [new_clip_queriesA[i] for i in idxA]
        clip_subqueryB = [new_clip_queriesB[i] for i in idxB]
        # with open('clickresultclip.txt', 'w') as fout:
        #     for query in clip_subqueryA:
        #         for doc in query.docs:
        #             if doc.click == 1:
        #                 fout.write("{}\n".format(1.0/doc.cost))
        prop_after_clip = get_prop_diagnostic(clip_subqueryA, clip_subqueryB)
        save_propfile(os.path.join(FLAGS.output_dict, rankerB_way), FLAGS.name_clip, clip_subqueryA, clip_subqueryB)

        n1, n2 = get_clickNo(subqueryA, subqueryB)
        clipbal_queriesA, clipbal_queriesB = clip_prop(bal_queriesA, bal_queriesB, 0.1)
        del_clipbal_queriesA = delete_noclick(clipbal_queriesA)
        del_clipbal_queriesB = delete_noclick(clipbal_queriesB)
        new_clipbal_queriesA = single_doc(del_clipbal_queriesA)
        new_clipbal_queriesB = single_doc(del_clipbal_queriesB)
        clipbal_subqueryA = [new_clipbal_queriesA[i] for i in idxA]
        clipbal_subqueryB = [new_clipbal_queriesB[i] for i in idxB]
        # with open('clickresultclipbal.txt', 'w') as fout:
        #     for query in clipbal_subqueryA:
        #         for doc in query.docs:
        #             if doc.click == 1:
        #                 fout.write("{}\n".format(1.0/doc.cost))
        prop_after_clipbal = get_prop_diagnostic(clipbal_subqueryA, clipbal_subqueryB)
        save_propfile(os.path.join(FLAGS.output_dict, rankerB_way), FLAGS.name_clipbal, clipbal_subqueryA, clipbal_subqueryB)

    end = timeit.default_timer()

    print("Finished building clicked logs.")
    print("Diagnostic checking: naive propensities: {}; balanced propensities: {}".format(prop_before, prop_after))
    print('Running time: {:.3f}s.'.format(end - start))

if __name__ == '__main__':
    main()
