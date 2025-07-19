from transformers import T5TokenizerFast
from transformers import BartTokenizer


class QA_Tokenizer_T5:
    def __init__(self, opts):
        self.tokenizer = T5TokenizerFast.from_pretrained(opts["tokenizer_name"])
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]
        self.vocab_size = self.tokenizer.vocab_size

    def get_tokenizer(self):
        return self.tokenizer

    def tokenize_question(self, question, context):
        question_tokenized = self.tokenizer(
            question,
            context,
            max_length=self.q_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        return question_tokenized

    def tokenize_answer(self, answer):
        answer_tokenized = self.tokenizer(
            answer,
            max_length=self.t_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        return answer_tokenized


class QA_Tokenizer_BART:
    def __init__(self, opts):
        self.tokenizer = BartTokenizer.from_pretrained(opts["tokenizer_name"])
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]

    def get_tokenizer(self):
        return self.tokenizer

    def tokenize_question(self, question, context):
        question_tokenized = self.tokenizer(
            question + context,
            max_length=self.q_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        return question_tokenized

    def tokenize_answer(self, answer):
        answer_tokenized = self.tokenizer(
            answer,
            max_length=self.t_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        return answer_tokenized
