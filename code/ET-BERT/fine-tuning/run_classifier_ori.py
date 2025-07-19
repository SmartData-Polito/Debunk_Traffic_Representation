"""
This script provides an exmaple to wrap UER-py for classification.
"""
import sys
sys.path.append('../ET-BERT')

import json
import random
import argparse
import torch
import torch.nn as nn
from uer.embeddings import *
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np
import time
import os

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)

        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        temp_output = output
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
            #return temp_output, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        # model.load_state_dict(torch.load(args.pretrained_model_path, map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}), strict=False)
        # 创建一个新的字典来存放重命名后的层名
        origin_model_params = torch.load(args.pretrained_model_path, map_location = args.device) 
        accuracy_model_params = {}

        # 遍历检查点中的所有键，并进行重命名
        for key, value in origin_model_params.items():
            if key == 'embedding.word_embedding.weight':
                new_key = 'embedding.word.embedding.weight'
            elif key == 'embedding.position_embedding.weight':
                new_key = 'embedding.pos.embedding.weight'
            elif key == 'embedding.segment_embedding.weight':
                new_key = 'embedding.seg.embedding.weight'
            else:
                new_key = key  # 如果其他层名没有问题，则保持不变
            accuracy_model_params[new_key] = value

        # 加载重命名后的检查点到模型中
        missing_keys, unexpected_keys = model.load_state_dict(accuracy_model_params, strict=False)
        print("Model has, but not in parameters:", missing_keys)
        print("Parameters have, but not in model", unexpected_keys)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
    
    # 如果参数被冻结
    print(args.frozen)
    if args.frozen:
        for param in model.parameters():
            param.requires_grad = False

        model.output_layer_1.weight.requires_grad = True
        model.output_layer_1.bias.requires_grad = True
        model.output_layer_2.weight.requires_grad = True
        model.output_layer_2.bias.requires_grad = True

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        print(f"Layer: {name} | Frozen: {not param.requires_grad}")

def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    # if torch.cuda.device_count() > 1:
    #     loss = torch.mean(loss)

    # if args.fp16:
    #     with args.amp.scale_loss(loss, optimizer) as scaled_loss:
    #         scaled_loss.backward()
    # else:
    #     loss.backward()
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()
    inference_time = 0
    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)

        start_time = time.time()
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        inference_time += time.time() - start_time

        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

        if print_confusion_matrix:
            print("Confusion Matrix:")
            print(confusion)

    precision = confusion.diag() / (confusion.sum(1) + 1e-9)
    recall = confusion.diag() / (confusion.sum(0) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    f1_score = f1.mean().item()

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    print("Precision: ", precision.mean().item())
    print("Recall: ", recall.mean().item())
    print("F1 Score: {:.4f}".format(f1_score))
    print("Inference time: {:.4f}".format(inference_time))

    return correct / len(dataset), precision.mean().item(), recall.mean().item(), f1_score, inference_time


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    # parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
    #                     help="Pooling type.")

    # parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
    #                     help="Specify the tokenizer."
    #                          "Original Google BERT uses bert tokenizer on Chinese corpus."
    #                          "Char tokenizer segments sentences into characters."
    #                          "Space tokenizer segments sentences into words according to space."
    #                          )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--frozen", action="store_true",
                        help="Whether frozens the embedding and pre-training layer")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Specify the dataset used for training.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(f"{args.train_path}/train.tsv")
    print("The number of labels:", args.labels_num)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_or_initialize_parameters(args, model)
    if torch.cuda.is_available():
        print("GPU OK")
    else:
        print("CPU OK")

    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, f"{args.train_path}/train.tsv")
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    
    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    #     args.amp = amp

    # if torch.cuda.device_count() > 1:
    #     print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)
    args.model = model

    result, best_result = 0.0, 0.0
    # -1 is used to mark which folder
    results_file_path = f"results/{args.dataset}/{args.train_path[-1]}_{args.learning_rate}_{args.frozen}_results.json"
    os.makedirs(f"results/{args.dataset}", exist_ok=True)
    os.makedirs(args.output_model_path, exist_ok=True)

    print("Start training.")
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        epoch_loss = 0.0
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in tqdm.tqdm(enumerate(batch_loader(batch_size, src, tgt, seg)), total=len(src)//batch_size):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            epoch_loss += loss.item()

        print("Epoch id: {}, Epoch loss: {:.3f}".format(epoch, epoch_loss / (len(src) // batch_size)))

        result = evaluate(args, read_dataset(args, f"{args.dev_path}/val.tsv"))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, f"{args.output_model_path}/finetuned_{args.dataset}_{args.learning_rate}_{args.frozen}_{args.train_path[-1]}.pth")
        
        results = {
            "epoch": epoch, 
            "loss": epoch_loss / (len(src) // batch_size), 
            "accuracy": result[0], 
            "precision": result[1], 
            "recall": result[2], 
            "f1_score": result[3],
            "inference_time": result[4] }

        with open(results_file_path, "a") as results_file:
            json.dump(results, results_file)
            results_file.write("\n")
        

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")

        model.load_state_dict(torch.load(f"{args.output_model_path}/finetuned_{args.dataset}_{args.learning_rate}_{args.frozen}_{args.train_path[-1]}.pth"))
        test_result = evaluate(args, read_dataset(args, f"{args.test_path}/test.tsv"), False)

        test_results = {
            "epoch": "test", 
            "loss": None, 
            "accuracy": test_result[0], 
            "precision": test_result[1], 
            "recall": test_result[2], 
            "f1_score": test_result[3],
            "inference_time": test_result[4] }
        
        with open(results_file_path, "a") as results_file:
            json.dump(test_results, results_file)
            results_file.write("\n")

if __name__ == "__main__":
    main()
