import gensim
import re
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from lib import config
_PAD = b"_PAD"
_UNK = b"_UNK"

_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def load_pretained_vector(word_vector_path):
    embedding_index = {}
    embedding_index["_PAD"] = np.zeros((config.EMBEDDING_DIM))
    embedding_index["_UNK"] = np.random.normal(size = (config.EMBEDDING_DIM))
    word_list = []
    model = gensim.models.Word2Vec.load(word_vector_path)
    word_vectors = model.wv

    for word,vocab_obj in model.wv.vocab.items():
        embedding_index[word] = word_vectors[word]
        word_list.append(word)
    num_words = len(word_list)

    word2index = {}
    if os.path.exists(config.VOCABULARY_PATH):
        with open(config.VOCABULARY_PATH,encoding='utf-8') as f:
            count = 0
            for line in f:
                word2index[line.strip()] = count
                count += 1
    else:
        for i in range(num_words):
            word2index[word_list[i]] = i
    #构造embedding矩阵
    embedding_matrix = np.zeros((num_words+2, config.EMBEDDING_DIM))
    for i in word2index.keys():
        embedding_matrix[word2index[i]] = embedding_index.get(i)

    return embedding_matrix,word_list

def data_to_token_ids(data_path, target_path, vocabulary_path):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary):
    """
    将sentence映射为word_index
    :param sentence: input sentence
    :param vocabulary: vocabulary of words2index
    :return:
    """
    words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]



def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
      if isinstance(space_separated_fragment, str):
        space_separated_fragment = space_separated_fragment.encode()
      words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w]


def create_vocabulary(vocabulary_path,word_list):
    if not gfile.Exists(vocabulary_path):
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            #在字典中添加_PAD,_UNK
            for i in _START_VOCAB:
                vocab_file.write(i + b"\n")
            for w in word_list:
                vocab_file.write(w.encode(encoding="utf-8") + b"\n")