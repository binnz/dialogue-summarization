from utils import make_train_data_from_txt
from config import Config
from transformers import BertTokenizer
import pickle


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(Config.pretrained_model_name_or_path)
    train_data = make_train_data_from_txt(Config, tokenizer)
    with open(f'{Config.pickle_path}', 'wb') as f:
        pickle.dump(train_data, f)
