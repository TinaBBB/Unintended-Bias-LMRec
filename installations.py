import os
from transformers import BertTokenizer


if __name__ == "__main__":
    """ Get tokenizer """
    # Save the slow pretrained tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)
