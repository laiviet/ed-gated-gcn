import corenlp
import os
import random
import numpy
import pickle
from data_utils import read_litbank_file, read_ace34_file, read_ace2_file
import tqdm
import multiprocessing
from constant import *

CORENLP_HOME = '/Users/vietld/tools/stanford-corenlp-full-2018-10-05'
os.environ['CORENLP_HOME'] = CORENLP_HOME

# We assume that you've downloaded Stanford CoreNLP and defined an environment
# variable $CORENLP_HOME that points to the unzipped directory.
# The code below will launch StanfordCoreNLPServer in the background
# and communicate with the server to annotate the sentence.

hostname = 'localhost'
# port = random.choice(list(range(9000, 9030)))

# hostname = 'localhost'
# port = 9000
# x = corenlp.CoreNLPClient(annotators="tokenize ssplit pos depparse".split(),
#                                start_server=False,
#                                endpoint='http://{}:{}'.format(hostname, 9000))

clients = [corenlp.CoreNLPClient(annotators="tokenize ssplit pos depparse".split(),
                                 start_server=False,
                                 endpoint='http://{}:{}'.format(hostname, port))
           for port in range(9000, 9010)]
properties = {
    'inputFormat': 'text',
    'outputFormat': clients[0].default_output_format,
    'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer',
    'tokenize.whitespace': 'true',
    'tokenize.language': 'Whitespace',
    'ssplit.eolonly': 'true'
}


ORI_ML = 100


def annotate(sentence):
    client = random.choice(clients)
    doc = client.annotate(sentence, properties=properties)
    # print(doc.sentence[0])
    # print('-'*80)
    return doc


def gen_graph(tokens):
    print(len(tokens))
    text = ' '.join(tokens)

    doc = annotate(text)
    # print(doc)
    sentences = doc.sentence
    assert len(sentences) == 1, '{} sentences, check {}'.format(len(sentences), text)

    l = len(tokens)

    tree = sentences[0].enhancedDependencies
    max_node = max([node.index for node in tree.node])
    matrix = numpy.eye(ORI_ML, dtype=int)
    assert max_node == l, 'More node than original,  check {}'.format(text)
    # assert l <= ORI_ML, 'Length exceed max len, l={}'.format(l)

    for edge in tree.edge:
        i = edge.source - 1
        j = edge.target - 1
        matrix[i][j] = 1
        matrix[j][i] = 1
    return matrix


def gen_litbank_wrapper(path):
    data = read_litbank_file(path)
    idx_graph = dict()
    for gid, tokens, _, _ in data:
        # try:

        idx_graph[gid] = gen_graph(tokens)
    # except:
    #     print(gid)
    #     print(len(tokens))
    #     print(' '.join(tokens))

    with open(path + '.graph', 'wb') as f:
        pickle.dump(idx_graph, f)


def gen_ace2_wrapper(path):
    print("Gen graph: ", path)
    cached = {}
    data = read_ace2_file(path)
    idx_graph = dict()
    get_from_cached = 0
    for gid, tokens, _, _, _ in data:
        if tuple(tokens) not in cached:
            g = gen_graph(tokens)
            cached[tuple(tokens)] = g
            idx_graph[gid] = g
            get_from_cached += 1
        else:
            idx_graph[gid] = cached[tuple(tokens)]

    with open(path + '.graph', 'wb') as f:
        pickle.dump(idx_graph, f)
    print(path, 'Cached: {}/{}'.format(get_from_cached, len(data)))


def gen_ace34_wrapper(path):
    print("Gen graph: ", path)
    cached = {}
    data = read_ace34_file(path)
    idx_graph = dict()
    get_from_cached = 0
    for gid, tokens, _, _, _ in data:
        if tuple(tokens) not in cached:
            g = gen_graph(tokens)
            cached[tuple(tokens)] = g
            idx_graph[gid] = g
            get_from_cached += 1
        else:
            idx_graph[gid] = cached[tuple(tokens)]

    with open(path + '.graph', 'wb') as f:
        pickle.dump(idx_graph, f)
    print(path, 'Cached: {}/{}'.format(get_from_cached, len(data)))


def gen_ace_wrapper(path):
    print("Gen graph: ", path)
    data = read_ace_file(path)
    idx_graph = dict()
    for gid, tokens, _, _ in data:
        g = gen_graph(tokens)
        idx_graph[gid] = g

    with open(path + '.graph', 'wb') as f:
        pickle.dump(idx_graph, f)
    print('Done: ', path)


def gen_all(path, fn):
    files = []
    train = os.path.join(path, 'train')
    dev = os.path.join(path, 'dev')
    test = os.path.join(path, 'test')
    files += sorted([os.path.join(train, x) for x in os.listdir(train) if x.endswith('tsv')])
    files += sorted([os.path.join(dev, x) for x in os.listdir(dev) if x.endswith('tsv')])
    files += sorted([os.path.join(test, x) for x in os.listdir(test) if x.endswith('tsv')])

    pool = multiprocessing.Pool(20)
    pool.map(fn, files)

    # for file in files:
    #     fn(file)


if __name__ == '__main__':
    # text = "A political and legal maelstrom has erupted after Pakistan 's president , Gen. Pervez Musharraf , unceremoniously suspended the country 's chief justice last week , in a step that lawyers and rights activists have called an assault on the independence of the judiciary ."
    # doc = annotate(text)

    import sys

    dataset = 'gpt3-cased'

    # assert dataset in ['ace-cased', 'ace34-cased', 'ace34-uncased', 'litbank-cased', 'litbank-uncased']



    # if dataset.startswith('ace34'):
    #     print('Gen graph for ACE34: ')
    #     gen_all(path='datasets/{}'.format(dataset), fn=gen_ace34_wrapper)
    # elif dataset.startswith('litbank'):
    #     print('Gen graph for LITBANK: ')
    #     gen_all(path='datasets/{}'.format(dataset), fn=gen_litbank_wrapper)
    # elif dataset.startswith('ace-'):
    #     print('Gen graph for ACE: ')
    #     gen_all(path='datasets/{}'.format(dataset), fn=gen_ace_wrapper)
    # elif dataset.startswith('gpt3'):
    #     print('Gen graph for GPT3: ')
    #     gen_all(path='datasets/{}'.format(dataset), fn=gen_litbank_wrapper)

    for i in range(14):
        gen_litbank_wrapper('datasets/litbank-cased/train/gpt-{}.tsv'.format(i))

