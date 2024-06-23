"""
XTuner 自定义数据集检查器

本脚本旨在验证和处理用于XTuner的自定义数据集，特别是针对Single-turn Fine-tuning (SFT)任务。
它执行以下几个关键功能：

1. 参数解析：接受配置文件路径作为命令行参数。
2. 配置加载：加载并解析指定的配置文件。
3. 数据集验证：检查数据集是否符合预期格式。
4. 数据处理：对数据集应用指定的转换和编码。
5. 格式转换：确保数据集格式与XTuner兼容。
6. 采样和打包：可选地对数据集进行采样和打包到指定长度。

该脚本特别适用于：
- 验证自定义数据集配置的正确性。
- 调试数据加载和处理流程。
- 确保数据集在训练前格式正确。

使用方法：
    xtuner check-custom-dataset /path/to/your/config.py

配置文件应指定以下详细信息：
- Dataset 路径和格式
- Tokenizer 设置
- 数据映射和模板函数
- 处理参数（最大长度，打包设置等）

这个工具是准备自定义数据集以用于XTuner的重要步骤，有助于在数据准备过程的早期发现和诊断问题。
"""

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from functools import partial

import numpy as np
from datasets import DatasetDict
from mmengine.config import Config

from xtuner.dataset.utils import Packer, encode_fn
from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(
        description='Verify the correctness of the config file for the '
        'custom dataset.')
    parser.add_argument('config', help='config file name or path.')
    args = parser.parse_args()
    return args


def is_standard_format(dataset):
    example = next(iter(dataset))
    if 'conversation' not in example:
        return False
    conversation = example['conversation']
    if not isinstance(conversation, list):
        return False
    for item in conversation:
        if (not isinstance(item, dict)) or ('input'
                                            not in item) or ('output'
                                                             not in item):
            return False
        input, output = item['input'], item['output']
        if (not isinstance(input, str)) or (not isinstance(output, str)):
            return False
    return True


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    tokenizer = BUILDER.build(cfg.tokenizer)
    if cfg.get('framework', 'mmengine').lower() == 'huggingface':
        train_dataset = cfg.train_dataset
    else:
        train_dataset = cfg.train_dataloader.dataset

    dataset = train_dataset.dataset
    max_length = train_dataset.max_length
    dataset_map_fn = train_dataset.get('dataset_map_fn', None)
    template_map_fn = train_dataset.get('template_map_fn', None)
    max_dataset_length = train_dataset.get('max_dataset_length', 10)
    split = train_dataset.get('split', 'train')
    remove_unused_columns = train_dataset.get('remove_unused_columns', False)
    rename_maps = train_dataset.get('rename_maps', [])
    shuffle_before_pack = train_dataset.get('shuffle_before_pack', True)
    pack_to_max_length = train_dataset.get('pack_to_max_length', True)
    input_ids_with_output = train_dataset.get('input_ids_with_output', True)

    if dataset.get('path', '') != 'json':
        raise ValueError(
            'You are using custom datasets for SFT. '
            'The custom datasets should be in json format. To load your JSON '
            'file, you can use the following code snippet: \n'
            '"""\nfrom datasets import load_dataset \n'
            'dataset = dict(type=load_dataset, path=\'json\', '
            'data_files=\'your_json_file.json\')\n"""\n'
            'For more details, please refer to Step 5 in the '
            '`Using Custom Datasets` section of the documentation found at'
            ' docs/zh_cn/user_guides/single_turn_conversation.md.')

    try:
        dataset = BUILDER.build(dataset)
    except RuntimeError:
        raise RuntimeError(
            'Unable to load the custom JSON file using '
            '`datasets.load_dataset`. Your data-related config is '
            f'{train_dataset}. Please refer to the official documentation on'
            ' `load_dataset` (https://huggingface.co/docs/datasets/loading) '
            'for more details.')

    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]

    if not is_standard_format(dataset) and dataset_map_fn is None:
        raise ValueError(
            'If the custom dataset is not in the XTuner-defined '
            'format, please utilize `dataset_map_fn` to map the original data'
            ' to the standard format. For more details, please refer to '
            'Step 1 and Step 5 in the `Using Custom Datasets` section of the '
            'documentation found at '
            '`docs/zh_cn/user_guides/single_turn_conversation.md`.')

    if is_standard_format(dataset) and dataset_map_fn is not None:
        raise ValueError(
            'If the custom dataset is already in the XTuner-defined format, '
            'please set `dataset_map_fn` to None.'
            'For more details, please refer to Step 1 and Step 5 in the '
            '`Using Custom Datasets` section of the documentation found at'
            ' docs/zh_cn/user_guides/single_turn_conversation.md.')

    max_dataset_length = min(max_dataset_length, len(dataset))
    indices = np.random.choice(len(dataset), max_dataset_length, replace=False)
    dataset = dataset.select(indices)

    if dataset_map_fn is not None:
        dataset = dataset.map(dataset_map_fn)

    print('#' * 20 + '   dataset after `dataset_map_fn`   ' + '#' * 20)
    print(dataset[0]['conversation'])

    if template_map_fn is not None:
        template_map_fn = BUILDER.build(template_map_fn)
        dataset = dataset.map(template_map_fn)

    print('#' * 20 + '   dataset after adding templates   ' + '#' * 20)
    print(dataset[0]['conversation'])

    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)

    if pack_to_max_length and (not remove_unused_columns):
        raise ValueError('We have to remove unused columns if '
                         '`pack_to_max_length` is set to True.')

    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            input_ids_with_output=input_ids_with_output),
        remove_columns=list(dataset.column_names)
        if remove_unused_columns else None)

    print('#' * 20 + '   encoded input_ids   ' + '#' * 20)
    print(dataset[0]['input_ids'])
    print('#' * 20 + '   encoded labels    ' + '#' * 20)
    print(dataset[0]['labels'])

    if pack_to_max_length and split == 'train':
        if shuffle_before_pack:
            dataset = dataset.shuffle()
            dataset = dataset.flatten_indices()
        dataset = dataset.map(Packer(max_length), batched=True)

        print('#' * 20 + '   input_ids after packed to max_length   ' +
              '#' * 20)
        print(dataset[0]['input_ids'])
        print('#' * 20 + '   labels after packed to max_length    ' + '#' * 20)
        print(dataset[0]['labels'])


if __name__ == '__main__':
    main()
