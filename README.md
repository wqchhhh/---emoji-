# ---emoji-
该模型是基于魔搭社区的三个大模型实现的，分别是人脸识别fer，中文StableDifusion-通用领域（文生图）模型，DCT-Net人像卡通化
实现了将输入的图片或利用文生图大模型进行关键词提取的图片，配上我们给出的，文本库.csv文件进行文字检索，配上文字输出表情包
以下是我们项目的流程图

<img width="325" alt="image" src="https://github.com/wqchhhh/---emoji-/assets/166973441/e8de9cc0-8487-444d-9036-0b70a3c0b6c2">

如图，
我们根据用户需求实现了生成表情包的功能，
让用户选择模型的模式

a）为导入图片
b）为文本生成图片

根据所选择的模式分为两类：

a）如果选择用户输入图片，模型读取图片，调用情绪识别模型进行情绪识别，然后将图片卡通化；
b）如果选择文生图，根据输入的文本生成图像，用情绪识别模型进行情绪识别；

根据识别的情绪，根据自定义的文本库，寻找一个合适的表情包文字，将文字与图片结合，输出表情包。

配置文件在requirem.txt

项目结果展示

第一个功能，根据用户输入图片

<img width="844" alt="a4597c689f14eebe251b23c83a5fccfd" src="https://github.com/wqchhhh/---emoji-/assets/166973441/f02d0fff-b875-48f8-b17d-64842ebc3010">


第二个功能：调用文生图模型

输入关键词：林黛玉

<img width="203" alt="image" src="https://github.com/wqchhhh/---emoji-/assets/166973441/c781cf2d-41fc-4ccb-9bcc-cc21b8ba4cc4">
<img width="209" alt="image" src="https://github.com/wqchhhh/---emoji-/assets/166973441/d99f06b0-5da2-403e-9b27-09a0d0c2dc77">

输入关键词：快乐的加菲猫

<img width="208" alt="image" src="https://github.com/wqchhhh/---emoji-/assets/166973441/8b5b0e8b-9742-482e-a288-102c1869c101">






