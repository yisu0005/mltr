from .click_log import *

'''
Translate original test file to the click data format
'''

def make_click_asrel(queries):
    newqueries = copy.deepcopy(queries)
    for query in newqueries:
        for doc in query.docs:
            doc.set_click(int(doc.rel))
    return newqueries


def save_testpropfile(path, name, queries):
    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path, name), 'w') as f:
        for query in queries:
            for doc in query.docs:
                if doc.click:
                    f.write(str(doc.click) + " " + "qid:" + str(query.clickid) + " " + "cost:" + str(1) + " " + str(
                        doc.feature))
                else:
                    f.write(str(doc.click) + " " + "qid:" + str(query.clickid) + " " + str(doc.feature))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_path', help='test path')
    parser.add_argument('output_dict', help='output directory')
    parser.add_argument('testfile_name', help='new test file name')
    FLAGS, unparsed = parser.parse_known_args()

    start = timeit.default_timer()
    test_queries = read_doc(FLAGS.test_path) ## FIXME: read_docs?
    clicked_query = make_click_asrel(test_queries)
    cleared_query = delete_noclick(clicked_query)
    single_queries = single_doc(cleared_query)
    save_testpropfile(FLAGS.output_dict, FLAGS.testfile_name, single_queries)
    end = timeit.default_timer()

    print("Finished building new test file.")
    print('Running time: {:.3f}s.'.format(end - start))

if __name__ == '__main__':
    main()
