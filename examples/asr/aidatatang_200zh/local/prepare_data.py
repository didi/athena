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
""" aidatatang_200zh dataset """

import os
import sys
import codecs
import pandas
from absl import logging

import tensorflow as tf
from athena import get_wave_file_length

SUBSETS = ['train', 'dev', 'test']

def convert_audio_and_split_transcript(directory, subset, out_csv_file):

    gfile = tf.compat.v1.gfile
    logging.info("Processing audio and transcript for {}".format(subset))
    audio_dir = os.path.join(directory, 'corpus', subset)
    trans_dir = os.path.join(directory, 'transcript')

    files = []
    char_dict = {}
    if not gfile.Exists(os.path.join(directory, subset)):
        data_file = os.path.join(directory, subset)
        # os.mkdir(data_file)
        os.makedirs(data_file, exist_ok=True)
        for filename in os.listdir(audio_dir):
            os.system("tar -zxvf" + audio_dir + filename + " -C" + data_file)

    with codecs.open(os.path.join(trans_dir, "aidatatang_200_zh_transcript.txt"), "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split(" ")
            wav_filename = items[0]
            labels = ""
            for item in items[1:]:
                labels += item
                if item in char_dict:
                    char_dict[item] += 1
                else:
                    char_dict[item] = 0
            files.append((wav_filename + ".wav", labels))

    files_size_dict = {}
    output_wav_dir = os.path.join(directory, subset)

    for root, subdirs, _ in gfile.Walk(output_wav_dir):
        for subdir in subdirs:
            for filename in os.listdir(os.path.join(root, subdir)):
                if filename.strip().split('.')[-1] != 'wav':
                    continue
                files_size_dict[filename] = (
                    get_wave_file_length(os.path.join(root, subdir, filename)),
                    subdir,
                )

    content = []
    for wav_filename, trans in files:
        if wav_filename in files_size_dict:
            filesize, subdir = files_size_dict[wav_filename]
            abspath = os.path.join(output_wav_dir, subdir, wav_filename)
            content.append((abspath, filesize, trans, subdir))
    files = content
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )
    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))




def processor(dircetory, subset, force_process):
    """ download and process"""
    if subset not in SUBSETS:
        raise ValueError(subset, 'is not in aidatatang')
    if force_process:
        logging.info("force process is set to be true")

    subset_csv = os.path.join(dircetory, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        logging.info("{} already exist".format(subset_csv))
        return subset_csv
    logging.info("Processing the AISHELL subset {} in {}".format(subset, dircetory))
    convert_audio_and_split_transcript(dircetory, subset, subset_csv)
    logging.info("Finished processing AISHELL subset {}".format(subset))
    return subset_csv

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    DIR = sys.argv[1]
    for SUBSET in SUBSETS:
        processor(DIR, SUBSET, True)