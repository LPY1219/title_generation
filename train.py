import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4'#最高的制定所用设备的代码，后面的制定都是相对于此
import torch
import random
import numpy as np
import argparse#命令行参数解析工具
import logging
from transformers.modeling_gpt2 import GPT2Config
from model import GPT2LMHeadModel
from transformers import BertTokenizer
from data_set import GPT2NewsTitleDataSet, collate_func
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# try:
#     from torch.utils.tensorboard import SummaryWriter #该包可以实现模型的可视化，随着训练轮数的增加，loss咋变这种
# except ImportError:
#     from tensorboardX import SummaryWriter   多卡训练的时候使用tensorboardX会导致文件重复生成错误
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
logger = logging.getLogger(__name__)
def train(model, device, train_data, args,device_):
    """
    训练模型
    Args:
        model: 模型
        device: 设备信息
        train_data: 训练数据类
        test_data: 测试数据类
        args: 训练参数配置信息
    Returns:

    """
    torch.cuda.set_device(args.local_rank) #必须放在最前面设置
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')#在使用 distributed 包的任何其他函数之前，需要使用 init_process_group 初始化进程组，同时初始化 distributed 包。
    #tb_write = SummaryWriter()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")
    # 计算真实的训练batch_size大小
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)#所谓梯度累积就是在积累了多个batch的梯度之后再进行参数更新
    #这实际上是变相加大了batch_size的大小，一般会取得更好的训练效果。为什么不直接调大batch_size呢？因为显卡的显存有限，没有办法一次装很大batch_size的数据
    train_sampler = DistributedSampler(train_data)# 如果使用Datadistributed必须使用这个sampler
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func,shuffle=False)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    if torch.distributed.get_rank() == 0:
        logger.info("总训练步数为:{}".format(total_steps))
    #model.cuda() ！！！！！！！！！！！为了加快运行速度进行并行训练，采用了如下并行方式
    # model = nn.DataParallel(model, device_ids=[int(i) for i in list(device_.replace(',',''))])
    model=model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)
    #device_ids需要按照列表的形式写进去
    #model = nn.DataParallel(model,device_ids=[0,1,2])
    #model.to(device)
    # 获取模型所有参数
    param_optimizer = list(model.module.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)#这个
    # 清空cuda缓存
    torch.cuda.empty_cache()
    # 将模型调至训练状态
    model.module.train()
    title_id = train_data.title_id
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    # 开始训练模型
    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):#显示进度条
        train_data_loader.sampler.set_epoch(iepoch+9)#进行分布式训练的时候为了保证每一轮的数据不同以及每一轮的数据分发到每一张卡上的数据相同必须用这种方式
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):#enumerate 用于同时列举数据和下标，形成一个元组
            input_ids = batch["input_ids"].cuda()
            #input_ids=batch['input_ids'].cuda(non_blocking=True)
            token_type_ids = batch["token_type_ids"].cuda()
            #token_type_ids = batch["token_type_ids"].cuda(non_blocking=True)
            # 获取训练结果
            outputs = model.module.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            #!!!!!!!!!!上面的代码做了改动，model.module()
            loss = outputs[0]
            tr_loss += loss.item()#X.item（）可以简单的理解为将张量转化为对应的数据的类型。但相比于直接用索引取值来讲，使用item方法的精度更高，所以
            #一般在计算损失函数时都使用item
            # 将损失值放到Iter中，方便观察
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())#在进度条前面实时动态显示损失值
            # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) #为防止梯度爆炸需要将梯度限制在某个阈值以内
            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # 如果步数整除logging_steps，则记录学习率和训练集损失值
                # if args.logging_steps > 0 and global_step % args.logging_steps == 0 and torch.distributed.get_rank() == 0:
                #     tb_write.add_scalar("lr", scheduler.get_lr()[0], global_step)
                #     tb_write.add_scalar("train_loss", (tr_loss-logging_loss) /
                #                         (args.logging_steps*args.gradient_accumulation_steps), global_step)
                #     logging_loss = tr_loss
                #如果步数整除eval_steps，则进行模型测试，记录测试集的损失
                # if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                #     eval_loss = evaluate(model, device, test_data, args)
                #     tb_write.add_scalar("test_loss", eval_loss, global_step)
                #     model.train()
        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        model_to_save = model.module if hasattr(model, "module") else model
        if torch.distributed.get_rank() == 0:
            model_to_save.save_pretrained(output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()



def evaluate(model, device, test_data, args):
    """
    对测试数据集进行模型测试
    Args:
        model: 模型
        device: 设备信息
        test_data: 测试数据类
        args: 训练参数配置信息

    Returns:

    """
    # 构造测试集的DataLoader
    if torch.distributed.get_rank()==0:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        logger = logging.getLogger(__name__)

    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func)
    if torch.distributed.get_rank() == 0:
        iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    title_id = test_data.title_id
    total_loss, total = 0.0, 0.0
    # 进行测试
    for step, batch in enumerate(iter_bar):
        # 模型设为eval  之所以这里要eval一下是为了删除模型中的dropout过程，显然在测试时不需要使用dropout
        model.eval()
        with torch.no_grad():#但下面的语句不需要进行反向传播时，就可以用这个语句，其表示在计算时不用保留计算图，因为计算图的构建就是为了便于误差反传。
            input_ids = batch["input_ids"].cuda()
            token_type_ids = batch["token_type_ids"].cuda()
            # 获取预测结果
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            loss = outputs[0]
            loss = loss.item()
            # 对loss进行累加
            total_loss += loss*len(batch["input_ids"])
            total += len(batch["input_ids"])
    # 计算最终测试集的loss结果
    test_loss = total_loss / total
    return test_loss




def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()#命令行参数解析工具，有了他，在命令行中就可以自由地添加参数了
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--device', default='0,1,2,3,4', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--config_path', default='./config/config.json', type=str, help='模型参数配置信息')
    parser.add_argument('--vocab_path', default='./vocab/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--train_file_path', default='./data_dir/train_data.json', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--test_file_path', default='./data_dir/test_data.json', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--pretrained_model_path', default='./model/checkpoint-99664', type=str, help='预训练的GPT2模型的路径')
    parser.add_argument('--data_dir', default='./data_dir', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=8, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=20, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=8, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='模型训练时的学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')
    parser.add_argument('--eval_steps', default=4000, type=int, help='训练时，多少步进行一次测试')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')#！！！！！！！！！感觉这里的路径有点问题
    parser.add_argument('--seed', type=int, default=2020, help='随机种子')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--title_max_len', type=int, default=32, help='生成标题的最大长度，要比max_len小')
    return parser.parse_args()



def main():
    # 设置模型训练参数
    args = set_args()
    # before your code runs
    # 设置显卡信息
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #CUDA应用运行时进行设备查询（比如deviceQuery）返回的设备ID可能与显卡实际的物理ID不一致
    #我们可以通过设置 CUDA_DEVICE_ORDER = PCI_BUS_ID 来要求运行时设备查询按照 PCI_BUS_ID 的顺序索引，从而使得 设备ID=物理ID 保证CUDA应用按期望使用指定设备
    # os.environ["CUDA_VISIBLE_DEVICES"] =args.device#_____________________________________
    # 获取device信息，用于模型训练
    #device = torch.device("cuda:1" if torch.cuda.is_available() and args.device != '-1' else "cpu")
    #USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda", args.local_rank)
    # device = torch.device("cuda:" + re.split(r",", args.device)[0] if USE_CUDA else "cpu")
    #device = torch.device("cuda" if USE_CUDA else "cpu")
    #torch.cuda.set_device('cuda:1')
    #device=torch.device("cuda" if torch.cuda.is)
    # 设置随机种子，方便模型复现
    if args.seed:
        torch.manual_seed(args.seed)#为cpu设置随机数种子
        random.seed(args.seed)#为random库设置随机种子
        np.random.seed(args.seed)
    # 加载模型的config
    model_config = GPT2Config.from_json_file(args.config_path)
    # 实例化GPT2LMHeadModel模型，这里我们没有加载预训练好的模型，而是直接从头开始训练。
    # 为什么从头开始训练？我们采用的是小模型，只有6层，并且词表也做了修改，没有找到合适的预训练模型。
    # 判断是否使用预训练好的GPT2模型
    if args.pretrained_model_path:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
    else:
        # 如果没有指定的预训练模型，则初始化模型
        model = GPT2LMHeadModel(config=model_config)
    # model = GPT2LMHeadModel(config=model_config)
    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)#do_lower_case 表示区分大小写
    # 将[space]作为一个分割整体，例如："我爱[Space]中国。"，使用原始tokenizer分词结果为"['我', '爱', '[', 'Space', ']', '中', '国', '。']";
    # 增加分割符号后的结果为"['我', '爱', '[Space]', '中', '国', '。']"
    tokenizer.add_tokens("[Space]", special_tokens=True)
    # 创建模型的输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 加载训练数据和测试数据
    train_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "train", args.train_file_path)
    #test_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "test", args.test_file_path)
    # 开始训练
    train(model, device, train_data, args,args.device)


if __name__ == '__main__':
    main()

