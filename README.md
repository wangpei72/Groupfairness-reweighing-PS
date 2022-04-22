# RQ5-reweighing算法思路梳理

经过文献的阅读，现在应该明白了reweighing的算法思路。

文章提出了三种预处理算法

- massaging - 此算法会更改数据集的标签
- reweighing - 此算法会给数据新增一列weight项重新训练，但是不会更改标签
- re-sampling  - 此算法虽然不会更改原来数据的标签，但是原先数据可能会因为重新采样而产生新的实例或者删除掉一些实例



准备采用上述后两者，其中resampling算法包括了 uniformly resampling（US） 和 preferential resampling（PS），PS需要使用到带有ranker（打分排序）的分类器用以判断被选为重复或者删除样例的标准，而前者不需要。

现在的技术思路：

AIF360显然提供了整个pipeline的实现，其中对数据进行reweighing的部分也有，但是重新训练的部分要看师姐给的代码。另外原来的文献也有给出源代码——https://sites.google.com/site/faisalkamiran/，这部分实验用的数据集有census，communities and crimes， Dutch 2001 census；有关我们也用到的数据集German的实验源码部分可以在 同作者的Classifying without discrimination一文中找到。

todo-list（任务拆解）：

- [x] 打开那该死的pycharm
- [x] 找到相关的notebook，然后找到使用reweighing的部分，先跑一遍
- [x] 然后将dataset的部分弄到手，也就是拿到带有weight的数据集
- [x] 你看看有没有resampling的部分，毕竟不更改数据维度不是更好吗
- [x] 打开re训练，基于初始模型重新训练应该做不到吧，因为带来了一项新的weight，也许只能resampling并且是PS，先试试吧，看来两种方式都要做，都成功了自然好，其一失败保留其二
- [x] 跑代码的时候配置服务器的环境，利用服务器硬件资源
- [x] 重新训练之后跑一边ckpt形式的神经元覆盖率

## 4-11问题list

dataset部分，四个权重基本算是按照4w全版本的样本分布情况计算出来了，adult（census）为例，取sex为attr

从上往下是优正、劣正，优负，劣负样本的weight数值，在RW算法中这个数值将扩充到data的新增一列

浅浅分析一下，优正，劣负样本小于1，表示这类样本个数多了，在sampling中将会被抹除部分实例；劣正，尤其优负样本大于1，这类样本个数太少了，sampling中将会duplicate引入重复的样本



![image-20220411224313528](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220411224313528.png)

dataset中的数据，34189是因为按照80%比例从原来的四万个数据集中抽取的train集

![image-20220411225150807](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220411225150807.png)



注意计算公平性指标时候，w（权重）的作用

![image-20220412141235980](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220412141235980.png)



正常情况下是全1矩阵



## 4-12新问题

aif360的data预处理方式与adf不样，vec的数值不同，原先的模型在这里也许整个要重来



晕，aif360的预处理方式是这样子的：

![image-20220412211620420](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220412211620420.png)

最后每个向量有18列，但实际上，原先的数据集只取了相当于4个特征，但是按照阶梯来分类了。据说这给预处理方式是取材于https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb这个地方的？

我想说，这个向量最后只有4个1，其余全0，如此稀疏，真的不会损失很多特征吗？（我是真不知道，但是感觉会损失）

而且这样下来，我们的处理也麻烦了，要给retrain喂的数据变得不对劲了，需要我们手动获取数据集的weight，我们可以知道那个weights的数值，手动加在正常的数据集上，然后训练。这下输入和输出维度均变为14（新增一列）

![image-20220412211643912](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220412211643912.png)

emm，或者，不增加weight，而是采用文章给出的PSampling，那么我们要操作的就更多了

- [ ] ~~先计算weights aif360帮你算了4w版本的权重数据，当然你也可以自己算~~
- [ ] ~~在计算当前D训练出的模型给出的打分（US不需要这部分）~~
- [ ] ~~当前模型再测试集中的NC数据，如果你用的还是32561的模型，不需要在计算，4w的话要重新算~~
- [ ] ~~对D进行四类样本的sampling，详见下图Algo5~~
- [ ] ~~获得$D_{ps}$~~
- [ ] ~~上述数据集重新训练~~
- [ ] ~~计算重新训练后的模型NC数据~~

现在已有的模型是基于32561的train集来的，那么想要4w份训练的那份，需要我们新开一个分支去操作，先用4w份的训练，再算NC（是不是有点重复），这个方案还是先不要去做

如果为了保持NC数据和32561那份的同步，那么后面步骤都需要自己算不依赖于360提供的实现，不然权重计算的结果是错误的

按照32561的打分，去做sampling，好的现在整理一下**最终实验思路**



- [ ] 计算原来模型的权重，手动实现，参照360实现思路
- [ ] 测试集得到NC数据
- [ ] predict得到原来模型的train集合打分-ranker获得
- [ ] 根据algo5给出的步骤sampling得到D_PS
- [ ] 用D_PS作为训练集重新训练模型
- [ ] 原测试集获得新模型的NC数据

![image-20220412213824776](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220412213824776.png)

有关这个权重计算的实现

![image-20220412214258391](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220412214258391.png)

计算权重就是fit函数↑ 很简单吧

之后是将数据集的weight赋值，这里是自己创造一个数组来接受所有的权重数据

![image-20220412223247410](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220412223247410.png)

然后就是说我们需要自己创建一个数组来存所有的weight，因为我们并没有用一个像360他自己用的dataset的对象来存储这些变量

期间要用到的所有conditonal_bool_vector用下述方法

![image-20220412223341398](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220412223341398.png)

condition_array的运用也需要解决，这个盘一下逻辑就好了

然后编码计算出四个数值，用1个四种权重加起来是（32561，1）形状的数组存好，之后还要有相应的分数记录，也是（32561，1）

之间可以用减法来确认一下是不是让spdifference变成0了

然后呢，原先的NC我们有了，接下来遍历train集，打分，记录一个ranker

之后就是algo5里展示的sampling方法，弄一个新的dataset——DPS

重新训练？or基于初始模型训练？

获得新的Classifier，然后得到新NC数据

## 4-13问题记录

![image-20220413155826271](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220413155826271.png)

目前32561版本的train集计算出来的weights

接下来要做的就是sampling的实现，sampling结束之后要做无偏估计证明？

看一下我之前存的site是怎么做的把



sites中的源码看了，已经不知道jar包和idea要怎么处理了，但是看了下ps的代码

![image-20220413202952472](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220413202952472.png)

emm，具体实现估计藏在Discrimination类中把，但是我的代码现在不能跳转。真的很难受，我尝试了maven安装3.8.6的weka，但是貌似不是安装这个就行的，他应该更像是在weka源码基础上开发了自己的约束，可以单独打包成一个可执行jar包

头疼，我现在该怎么做呢，他为什么总是爆红呢

在这个里面添加了一个lib不知道行不行哦

![image-20220413204522631](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220413204522631.png)

看样子是不行，因为这是lib下的jar代码结构

![image-20220413204659359](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220413204659359.png)

这个和给出的本身工程不久一样了吗，说明给出的sourcecode果然是按照约束重新开发的weka版本

所以我们应该仔细读一下readme，里面貌似只说了如何run实验，但是我们现在应该做的

- [ ] 是复现PS，所以只要手动跳转代码，参考PS流程（伪代码已经有了其实不难开发，看看人家实现思路是主要的），
- [ ] 然后用python复现，测一下公平指标，
- [ ] 完了以后上训练了，
- [ ] 训练完了以后在计算NC

## 4-14

我受够了

今天他娘的一定要做完，累死我了

![image-20220412213824776](C:\Users\wpkks\AppData\Roaming\Typora\typora-user-images\image-20220412213824776.png)

再来一遍这个伪代码

从第八行开始啊，首先我们四个权重：
$$
W(b,+) = 2.19 劣正DP\\
W(b,-) = 0.85劣负DN\\
W(w,+) = 0.79优正FP\\
W(w,-) = 1.09 优负FN\\
$$
DP - Deprived Positive(劣势正样本)   DN - Deprived Negative（劣势负样本）

FP - Favored Positive（优势正样本） FN - Favored Negative（优势负样本） 

除此之外，我们还有一份ranker，该ranker按照正样本分类probability，==升序排列DP和FP的正样本==，==降序排列DN和FN的负样本==

这样子安排是为确保在ranker中的data object，越早出现，表示他越接近用以sampling的boundary（这里结合图有点难理解）

PS从原始的训练集开始，迭代地duplicates（DP和FN集合中的）并且removes objects（DN和FP中的），按照以下方式：

- 总是removing 最接近于boundary的data objects，以减少一个group的大小， 例如：ranked list中的top元素
- 复制最靠近boundary的data objects以增加sample的大小，当一个object被重复了一次，他将被移动（和他的复制品一起），移动到ranking的bottom，重复这个步骤知道达到预期数目的objects的复制。

伪代码的翻译：结合实际数值以后：

9和13的公式总感觉怪怪的啊，我觉得不仅13的权重给错了，取整括号的顺序也错了

8： 将2份DP（劣势正样本）的复制增加到D_ps，

9：增加（2.19 - 2*|DP|）的向下取整结果 个lowest-排序的（从bottom往top取）元素的DP（劣正）到D_ps

10: 增加 （0.85*|DN|）的向下取整结果 个lowest-ranked 元素的DN（劣负）到D_ps(==换言之删除掉最靠近boundary（负样本概率较大的那边)的0.15%|DN|==）

11: 增加（0.79*|FP|）的向下取整结果个 highest-ranked 元素的FP（优正）到D_ps (==换言之删除掉最靠近boundary（负样本概率较大的那边)的0.21%|FP|==）

12：增加 1 份FN（优负）的复制到D_ps

13：增加（1.09 - 1）*|FN|的向下取整结果个 highest-ranked 元素的FN（优负）增加到D_ps(==换言之循环置底的方式重复最靠近boundary（正样本概率较低的那边)的0.15%|DN|==）



