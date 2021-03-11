import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import json

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def evaluate(args, model, tokenizer, prefix=""):

    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!

    all_results = []
    # start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            # if args.model_type in ["xlnet", "xlm"]:
            #     inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
            #     # for lang_id-sensitive xlm models
            #     if hasattr(model, "config") and hasattr(model.config, "lang2id"):
            #         inputs.update(
            #             {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
            #         )

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [output[i].tolist() for output in outputs.to_tuple()]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # evalTime = timeit.default_timer() - start_time

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results

def load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True):
        processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

        examples = processor.get_dev_examples('', filename=args.file_name)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt"
        )
        if output_examples:
            return dataset, examples, features
        return dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="bert", help="")
parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="")
parser.add_argument("--max_query_length", type=int, default=30, help="")
parser.add_argument("--max_seq_length", type=int, default=300, help="")
parser.add_argument("--doc_stride", type=int, default=100, help="")
parser.add_argument("--max_answer_length", type=int, default=15, help="")
parser.add_argument("--version_2_with_negative", action="store_true", help="")
parser.add_argument("--output_dir", type=str, default="", help="")
parser.add_argument("--file_name", type=str, default="try.json", help="")
parser.add_argument("--config_name", default="", type=str, help="")
parser.add_argument("--cache_dir",default="",type=str,help="")
parser.add_argument("--tokenizer_name",default="",type=str,help="")
parser.add_argument("--do_lower_case", action="store_true", help="")
parser.add_argument("--eval_batch_size", default=8, type=int, help="")
parser.add_argument("--n_best_size",default=15,type=int,help="")
parser.add_argument("--overwrite_cache", action="store_true")
parser.add_argument("--verbose_logging",action="store_true")
parser.add_argument("--null_score_diff_threshold",type=float,default=0.0)

args = parser.parse_args([])

def run_prediction(sent1,sent2,model):

    # prepare data file
    pred={
        "version": "dev",
        "data": [{
            "title": "doc-1",
            "paragraphs": [{
                "qas": [{
                    "question": sent2,
                    "answers": [],
                    "id": "1",
                    "is_impossible": True
                    }],
                "context": sent1
            }]}]}
    with open(args.file_name,'w') as f:
        json.dump(pred,f)

    # Load pretrained model and tokenizer
    device = torch.device("cpu")
    args.device = device
    if model=='bert-pretrained-squad2.0':
        args.model_name_or_path='../models/squad_bert_pretrained_try_1/checkpoint-1500/'
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,
        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)


    # Evaluate
    result = evaluate(args, model, tokenizer)
    print(result)

    # post processing
    with open(args.output_dir+"nbest_predictions_.json",'r') as f:
        preds=json.load(f)["1"]

    return preds

