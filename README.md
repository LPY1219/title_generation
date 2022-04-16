# title_generation
根据文本生成标题，最终生成质量较差，但跑通了整个流程，可以供初学者参考
安装如下第三方库：
gevent == 1.3a1
flask == 0.12.2
transformers == 3.0.2
pytorch >= 1.4
tqdm
numpy


链接：https://pan.baidu.com/s/1RzOzQ_nNNx5SjSHMxu_YiA?pwd=1327 
提取码：1327
将链接中的模型下载后放到同目录下
‘’
运行new_http.py 待界面中出现load model ending即可进行新闻标题生成。
在本地测试直接在浏览器中输入"127.0.0.1:5555/news-title-generate"，如果给他人访问，只需将"127.0.0.1"替换成本机电脑的IP地址即可。
“

各个文件夹说明：
config文件存储GPT-2网络的结构信息
data_dir存储训练和测试好的数据
event是tensorboaed记录的训练过程的可视化结果
image是可视化后的图片
model存储这训练好的模型
static、template存储为html网页建造的模板
vocab存储分词的词表
generate_title.py是用来进行文本标题生成的代码
model.py存储着修改后的模型
new_http.py启动网页
train.py是用来进行模型训练的代码
data_helper.py和date_set.py是用来进行数据处理的

本项目为本人的课程大作业，主要参考了刘聪大神的开源项目
@misc{GPT2-NewsTitle,
  author = {Cong Liu},
  title = {Chinese NewsTitle Generation Project by GPT2},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  url="https://github.com/liucongg/GPT2-NewsTitle",
}
