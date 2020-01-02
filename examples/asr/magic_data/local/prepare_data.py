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
""" magic_data dataset """

import os
import sys
import codecs
import pandas
from absl import logging

import tensorflow as tf
from athena import get_wave_file_length

SUBSETS = ["train", "dev"]

def convert_audio_and_split_transcript(directory, subset, out_csv_file):

    gfile = tf.compat.v1.gfile
    logging.info("Processing audio and transcript for {}".format(subset))
    audio_dir = os.path.join(directory, subset)
    trans_dir = os.path.join(directory, subset, "TRANS.txt")

    files = []
    with codecs.open(trans_dir,'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            items = line.strip().split('\t')
            wav_filename = items[0]
            labels = items[2]
            speaker = items[1]
            files.append((wav_filename, speaker, labels))
    files_size_dict = {}
    for root, subdirs, _ in gfile.Walk(audio_dir):
        for subdir in subdirs:
            for filename in os.listdir(os.path.join(root, subdir)):
                files_size_dict[filename] = (
                    get_wave_file_length(os.path.join(root, subdir, filename)),
                    subdir,
                )
    content = []
    for wav_filename, speaker, trans in files:
        if wav_filename in files_size_dict:
            filesize, subdir = files_size_dict[wav_filename]
            abspath = os.path.join(audio_dir, subdir, wav_filename)
            content.append((abspath, filesize, trans, speaker))

    files = content
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )
    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))

def processor(dircetory, subset, force_process):
    """ download and process """
    if subset not in SUBSETS:
        raise ValueError(subset, "is not in magic_data")
    if force_process:
        logging.info("force process is set to be true")

    subset_csv = os.path.join(dircetory, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        logging.info("{} already exist".format(subset_csv))
        return subset_csv
    logging.info("Processing the magic_data subset {} in {}".format(subset, dircetory))
    convert_audio_and_split_transcript(dircetory, subset, subset_csv)
    logging.info("Finished processing magic_data subset {}".format(subset))
    return subset_csv

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    DIR = sys.argv[1]
    for SUBSET in SUBSETS:
        processor(DIR, SUBSET, True)

