class Config:
    pretrained_model_name_or_path = 'hfl/rbtl3'

    # model params
    num_epoch = 1
    dim_model = 1024
    num_heads = 12
    dim_ff = 1024
    dropout = 0.1
    num_layers = 3
    max_src_num_length = 128
    max_utter_num_length = 128
    max_decode_output_length = 256
    utter_type = 3
    vocab_size = 21128
    max_grad_norm = 1.0

    # train
    seed = 123
    device = 'cpu'
    use_pickle = True
    data_dir = './data'
    train_data_path = f'{data_dir}/AutoMaster_TrainSet.csv'
    predict_data_path = f'{data_dir}/AutoMaster_TestSet.csv'
    predict_output = 'prediction_result'
    lr = 1e-5
    fn = 'ckpt'
    load = False
    pickle_path = f'{data_dir}/train_data.pkl'
    predict_pickle_path = f'{data_dir}/predict_data.pkl'
    betas = (0.9, 0.98)
    batch_size = 2
