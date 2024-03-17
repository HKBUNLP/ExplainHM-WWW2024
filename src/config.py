from sacred import Experiment

ex = Experiment("Meme", save_git_info=False)


def _loss_names(d):
    ret = {
        "clf": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "Meme"
    seed = 0
    datasets = ["meme"]
    loss_names = _loss_names({"clf": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    temperature = 0.05

    # Image setting
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    image_size = 224
    patch_size = 16

    # Text Setting
    max_text_len = 40
    tokenizer = "t5-small"
    vocab_size = 32128
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.3

    # Transformer Setting
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'google/vit-base-patch32-224-in21k'
    hidden_size = 768
    num_heads = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    get_recall_metric = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32
    # resume_from = ""

@ex.named_config
def task_train():
    exp_name = "MEME"
    datasets = ["meme"]
    loss_names = _loss_names({
        "clf": 1,
    })
    batch_size = 256
    temperature = 0.05
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    whole_word_masking = False

    vocab_size = 32128
    max_text_len = 40
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    learning_rate = 5e-5
    val_check_interval = 1.0
    hidden_size = 768
    num_heads = 12


# visual encoder
@ex.named_config
def vit32_base224():
    vit = "google/vit-base-patch32-224-in21k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

@ex.named_config
def vit16_base224():
    vit = "google/vit-base-patch16-224-in21k"
    patch_size = 16
    image_size = 224
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

@ex.named_config
def vit16_base384():
    vit = "google/vit-base-patch16-384"
    patch_size = 16
    image_size = 384
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

@ex.named_config
def clip32_base224():
    vit = "openai/clip-vit-base-patch32"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

@ex.named_config
def clip16_base224():
    vit = "openai/clip-vit-base-patch16"
    patch_size = 16
    image_size = 224
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

# text encoder
@ex.named_config
def text_bert():
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    input_text_embed_size = 768

@ex.named_config
def text_roberta_large():
    tokenizer = "roberta-large"
    vocab_size = 50265
    input_text_embed_size = 1024

# text encoder
@ex.named_config
def text_t5_small():
    tokenizer = "google/flan-t5-small"
    vocab_size = 32128
    input_text_embed_size = 512

@ex.named_config
def text_t5_base():
    tokenizer = "google/flan-t5-base"
    vocab_size = 32128
    input_text_embed_size = 768

@ex.named_config
def text_t5_large():
    tokenizer = "google/flan-t5-large"
    vocab_size = 32128
    input_text_embed_size = 1024

# random augmentation
@ex.named_config
def vit_randaug():
    train_transform_keys = ["vit_randaug"]
