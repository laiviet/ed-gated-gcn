import corenlp
import os
import pickle
from data_utils import read_token_from_file

CORENLP_HOME = '/Users/vietld/tools/stanford-corenlp-full-2018-10-05'
os.environ['CORENLP_HOME'] = CORENLP_HOME

# We assume that you've downloaded Stanford CoreNLP and defined an environment
# variable $CORENLP_HOME that points to the unzipped directory.
# The code below will launch StanfordCoreNLPServer in the background
# and communicate with the server to annotate the sentence.



with corenlp.CoreNLPClient(annotators="tokenize ssplit pos depparse".split(),
                           start_server=False) as client:
    properties = {
        'inputFormat': 'text',
        'outputFormat': client.default_output_format,
        'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer',
        'tokenize.whitespace': 'true',
        'tokenize.language': 'Whitespace'
    }


def annotate(sentence):
    print('-' * 80)
    doc = client.annotate(sentence, properties=properties)
    print(sentence)
    print(len(doc.sentence))
    return doc.sentence


def gen_graph(path):
    data = read_token_from_file(path)
    graphs = {}
    for sample_id, tokens in data:
        print(sample_id)
        sentences = annotate(' '.join(tokens))
        total_length = 0
        for sentence in sentences:
            max_node = max([x.index for x in sentence.enhancedPlusPlusDependencies.node])
            total_length += max_node
        assert total_length == len(tokens)

        offset = 0
        A = []
        for i in range(total_length):
            A.append([0] * total_length)

        roots = []
        for sentence in sentences:
            tree = sentence.enhancedPlusPlusDependencies
            roots.append(offset + tree.root[0] - 1)
            max_node = max([x.index for x in tree.node])
            for _, edge in enumerate(tree.edge):
                i = offset + edge.source - 1
                j = offset + edge.target - 1
                # print(i, j)
                A[i][j] = 1
                A[j][i] = 1
            offset += max_node
        for i in roots:
            for j in roots:
                if i != j:
                    A[i][j] = 1
                    A[j][i] = 1
        graphs[sample_id] = A
    output_file = '{}.graph'.format(path)
    with open(output_file, 'wb') as f:
        pickle.dump(graphs, f)


def gen_all_graph():
    # gen_graph('datasets/litbank/dev')
    # gen_graph('datasets/litbank/train')
    # gen_graph('datasets/litbank/test')

    # gen_graph('datasets/ace2-uncased/train')
    # gen_graph('datasets/ace2-uncased/dev')
    # gen_graph('datasets/ace2-uncased/test')

    gen_graph('datasets/ace-cased/train')
    gen_graph('datasets/ace-cased/dev')
    gen_graph('datasets/ace-cased/test')




def test_get_dist_to_target():
    f = open('datasets/litbank/dev.graph', 'rb')
    data = pickle.load(f)
    f.close()

    keys = list(data.keys())
    idx = 2
    print(keys[idx])
    a = data[keys[idx]]
    print(len(a))
    for i in a:
        print(i)


if __name__ == '__main__':
    gen_all_graph()
