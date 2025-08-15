# TreeSeqCC

### Requirements

+ pandas 1.5.3
+ numpy 1.26.4
+ numba 0.55.2
+ tensorflow >=2.7.0
+ keras >=2.7.0
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE could be reconfigured by user
+ [ASTSDL](https://github.com/yuyaoshen/ASTSDL)
  + 基于抽象语法树的编码序列，用于预先对代码功能进行基于语法结构的预测，生成代码功能预测的语义向量AstFP
+ [CodeSDL](https://github.com/yuyaoshen/CodeSDL)
  + 基于代码文本的token序列，用于预先对代码功能进行基于文本词性的预测，生成代码功能预测的语义向量SrcFP

### Usage
| 文件            | 用途                                                         |
| :-------------- | :----------------------------------------------------------- |
| checkVersion.py | 用于检测环境是否一致，避免包依赖产生的错误。                 |
| TreeSeqCC.py    | Code Clone Detection主程序，提供两种检测模式detectionType="Free"/"Sample-based"："Free"模式检测待测数据集中所有代码片段之间的相似性；"Sample-based"模式仅检测"samples"文件夹中给定样本对的相似度。FP_Sim_Threshold为功能相似性阈值，默认设定为0.9，VON_Sim_Threshold为节点类型相似性阈值，默认设定为0.9。 |

#### 相关文献：

1. [ASTENS-BWA: Searching partial syntactic similar regions between source code fragments via AST-based encoded sequence alignment](https://www.sciencedirect.com/science/article/abs/pii/S0167642322000727)
2. [ASTSDL: Predicting the Functionality of Incomplete Programming Code via AST-Sequence-based Deep Learning Model](http://engine.scichina.com/doi/10.1007/s11432-021-3665-1)
