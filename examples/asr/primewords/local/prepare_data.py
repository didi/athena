# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
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
""" primewords dataset """

import os
import sys
import codecs
import pandas
from absl import logging

import tensorflow as tf
# from athena import get_wave_file_length

def convert_audio_and_split_transcript(directory, out_csv_file):

    gfile = tf.compat.v1.gfile
    logging.info("Processing audio and transcript for {}".format('primewords'))
    audio_dir = os.path.join(directory, "audio_files")
    trans_dir = os.path.join(directory, "set1_transcript.json")

    files = []
    with codecs.open(trans_dir, 'r', encoding='utf-8') as f:
        items = eval(f.readline())
        for item in items:
            wav_filename = item['file']
            # labels = item['text']

            labels = ''.join([x for x in item['text'].split()])
            # speaker = item['user_id']
            files.append((wav_filename, labels))

    files_size_dict = {}
    for subdir in os.listdir(audio_dir):
        for root, subsubdirs, _ in gfile.Walk(os.path.join(audio_dir, subdir)):
            for subsubdir in subsubdirs:
                for filename in os.listdir(os.path.join(root, subsubdir)):
                    files_size_dict[filename] = (
                        os.path.join(root, subsubdir, filename),
                        root,
                        subsubdir
                    )
                    # print(os.path.join(root,subsubdir,filename))
    content = []
    for wav_filename, trans in files:
        if wav_filename in files_size_dict:
            filesize, root, subsubdir = files_size_dict[wav_filename]
            abspath = os.path.join(root, subsubdir, wav_filename)
            content.append((abspath, filesize, trans, None))
    files = content
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )

    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))

def processor(dircetory, force_process):
    if force_process:
        logging.info("force process is set to be true")

    subset_csv = os.path.join(dircetory, 'train' + ".csv")
    if not force_process and os.path.exists(subset_csv):
        logging.info("{} already exist".format(subset_csv))
        return subset_csv
    logging.info("Processing the primewords subset {} in {}".format('train', dircetory))
    convert_audio_and_split_transcript(dircetory, subset_csv)
    logging.info("Finished processing primewords subset {}".format('train'))
    return subset_csv

if __name__ == '__main__':

    logging.set_verbosity(logging.INFO)
    DIR = sys.argv[1]
    processor(DIR,True)

