import logging
from config import Config
import torch
from model import build_model
from transformers import BertTokenizer
from utils import eval_test

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logging.info('*** Initializing Predict Mode ***')
    epoch = 0
    device = torch.device(Config.device)
    tokenizer = BertTokenizer.from_pretrained(Config.pretrained_model_name_or_path)
    model = build_model(Config).to(device)
    state_dict = torch.load(f'{Config.data_dir}/{Config.fn}_{epoch}.pth')
    model.load_state_dict(state_dict['model'])
    model.eval()
    qa = "09款 经典轩逸1.6手动  ，14万公里  亮了故障灯，说是三元催化的问题，请问要紧吗？用了两瓶碳无敌 不知是否有效？"
    dialogue = "技师说：你好，建议把三元催化拆出来用清水洗一下吧！|车主说：是因为什么原因需要清洗呢？洗了以后有效果吗？|技师说：把里面的碳清冼一下，三元催化干净了当然有效果了|车主说：好 ，店里的师傅说  不一定有效果  建议换掉！是忽悠吗？|技师说：是的，先清冼后最做下一部打算吧，看看清冼后能不能用嘛！|技师说：用水洗了最用气枪把水吹一下就可以了。|技师说：电脑检测是报什么故障的，是三元催化器还是热氧传感器？|车主说：好|车主说：报的是三元催化故障|技师说：那就清冼三元了！|车主说：如果暂时不洗   照常开上路 会影响什么吗？|技师说：这个无什么影响的，放心使用！|车主说：好的 谢谢了！"
    real_report = "你好把三元催化拆出来用清洗一下吧"
    text = eval_test(Config, qa, dialogue, tokenizer, model, device)
    print('qa: {}'.format(qa))
    print('dialogue: {}'.format(dialogue))
    print('report: {}'.format(text))
    print('real report: {}'.format(real_report))
    print("eval test finish")
