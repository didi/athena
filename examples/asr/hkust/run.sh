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

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

source tools/env.sh

stage=0
stop_stage=100

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Preparing data"
    # tar -xvf /nfs/project/datasets/opensource_data/hkust/LDC2005S15.tar
    # tar -xvf /nfs/project/datasets/opensource_data/hkust/LDC2005T32.tar
    python examples/asr/hkust/local/prepare_data.py /nfs/project/datasets/opensource_data/hkust
    mkdir -p examples/asr/hkust/data
    cp /nfs/project/datasets/opensource_data/hkust/{train,dev}.csv examples/asr/hkust/data/

    # cal cmvn
    cat examples/asr/hkust/data/train.csv > examples/asr/hkust/data/all.csv
    tail -n +2 examples/asr/hkust/data/dev.csv >> examples/asr/hkust/data/all.csv
    python athena/cmvn_main.py examples/asr/hkust/mpc.json examples/asr/hkust/data/all.csv

    # prepare language model
    tail -n +2 examples/asr/hkust/data/train.csv | cut -f 3 > examples/asr/hkust/data/text
    # tools/kenlm/build/bin/lmplz -o 4 < examples/asr/hkust/data/text > examples/asr/hkust/data/4gram.arpa
    # TODO: prepare the code for training n-gram model
    tail -n +2 examples/asr/hkust/data/train.csv | awk '{print $3"\t"$3}' > examples/asr/hkust/data/train.trans.csv
    tail -n +2 examples/asr/hkust/data/dev.csv | awk '{print $3"\t"$3}' > examples/asr/hkust/data/dev.trans.csv
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # pretrain stage
    python athena/main.py examples/asr/hkust/mpc.json
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # finetuning stage
    python athena/main.py examples/asr/hkust/mtl_transformer.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # decoding stage
    python athena/decode_main.py examples/asr/hkust/mtl_transformer.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # decoding stage
    python athena/main.py examples/asr/hkust/rnnlm.json
fi

