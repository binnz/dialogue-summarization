import logging
import pandas as pd
from utils import evaluate
from config import Config
import torch
import pickle
from model import build_model
from transformers import BertTokenizer
from utils import make_predict_data_from_txt, DialogDataset, BalancedDataLoader, Batch

logging.basicConfig(level=logging.INFO)

"""
set the batch size to 1
"""
if __name__ == '__main__':
    logging.info('*** Initializing Predict Mode ***')
    epoch = 0
    device = torch.device(Config.device)
    tokenizer = BertTokenizer.from_pretrained(Config.pretrained_model_name_or_path)

    if Config.use_pickle:
        with open(f'{Config.predict_pickle_path}', 'rb') as f:
            predict_data = pickle.load(f)
    else:
        predict_data = make_predict_data_from_txt(Config, tokenizer)
        with open(f'{Config.predict_pickle_path}', 'wb') as f:
            pickle.dump(predict_data, f)
    dataSet = DialogDataset(predict_data, tokenizer)
    data_loader = BalancedDataLoader(dataSet, tokenizer.pad_token_id, 'predict')

    model = build_model(Config).to(device)
    state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth', map_location=device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    print('*** Start Evaluate ***')
    prediction = []
    qids = []
    total_nums = len(data_loader)
    for i, data in enumerate(data_loader):
        batch = Batch(data, device, mode='eval', pad=tokenizer.pad_token_id)
        text = evaluate(batch, tokenizer, model, device)
        prediction.append(text)
        qids.append(batch.qid[0])
        print(text)
        print("predict finish. QID is {}".format(batch.qid[0]))
    dataFrame = pd.DataFrame({'QID': qids, 'Prediction': prediction})

    dataFrame.to_csv(f'{Config.data_dir}/{Config.predict_output}.csv', index=True, sep=',')
    print("evaluate success. total num:{}".format(total_nums))
