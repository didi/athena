# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" segment word """

import sys
import re
from absl import logging
import jieba


def segment_trans(vocab_file, text_file):
    ''' segment transcripts according to vocab
        using Maximum Matching Algorithm
    Args:
      vocab_file: vocab file
      text_file: transcripts file
    Returns:
      seg_trans: segment words
    '''
    jieba.set_dictionary(vocab_file)
    with open(text_file) as TEXT:
      lines = TEXT.readlines()
      sents = ''
      for line in lines:
          sents += line
      words = jieba.cut(sents, HMM=False)
      seg_words = ' '.join(words)
      seg_words = re.sub(r'\n ', r'\n', seg_words)
      return seg_words


if __name__ == "__main__":
    if len(sys.argv) < 3:
      logging.warning('Usage: python {} vocab_file text_file'.format(sys.argv[0]))
      exit(1)
    logging.set_verbosity(logging.INFO)
    vocab_file = sys.argv[1]
    text_file = sys.argv[2]
    seg_trans = segment_trans(vocab_file, text_file)
    print(seg_trans)