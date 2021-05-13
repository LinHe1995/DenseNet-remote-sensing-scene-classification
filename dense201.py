import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple

#python列表，可变序列
__all__ = ['DenseNet', 'densenet201']

#python字典，可变容器0
model_urls = {
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}

#nn.Module
class _DenseLayer(nn.Module):

    #构造方法
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))#in_channels=num_input_features, out_channels=bn_size * growth_rate
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    # 做bn_func nrc1
    def bn_function(self, inputs: List[Tensor]) -> Tensor:#标记返回值数据类型tensor
        concated_features = torch.cat(inputs, 1)#torch.cat将两个张量拼接在一起
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features))) # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    # 如果需要为这个张量计算梯度,那么该函数返回True,否则返回False.
    # 如果一个张量我们需要为它计算梯度,这并不意味着这个张量地grad属性会一直被保持,详细信息请参考is_leaf属性.
    #     如果一个张量地requires_grad=True,那么在调用backward()方法时反向传播计算梯度,我们
    #     会为这个张量计算梯度,但是计算完梯度之后这个梯度并不一定会一直保存在属性grad中.只有对于
    #     requires_grad=True的叶子张量,我们才会将梯度一直保存在该叶子张量的grad属性中,对于非叶子节点,
    #     即中间节点的张量,我们在计算完梯度之后为了更高效地利用内存,我们会将梯度grad的内存释放掉.
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # 做bn_func并重新运行一次检查点部分的前向传播 nrc1
    # 作为检查点的部分不再保存中间激活值，而是在反向传播中重新计算一次中间激活值，即重新运行一次检查点部分的前向传播。
    # checkpoint操作在训练过程中可以节省下作为检查点部分所需占的内存，但是要付出在反向传播中重新计算的时间代价。
    @torch.jit.unused
    # This decorator indicates to the compiler that a function or method should be ignored and
    # replaced with the raising of an exception. This allows you to leave code in your model
    # that is not yet TorchScript compatible and still export your model.
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method # noqa: T484
    def forward(self, input: List[Tensor]) -> Tensor:
        pass # 不做任何事情，一般用做占位语句

    @torch.jit._overload_method # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor): # isinstance() 函数来判断一个对象是否是一个已知的类型
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT") # 在引发指定类型的异常的同时，附带异常的描述信息

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features) # 做bn_func并重新运行一次检查点部分的前向传播 nrc1
        else:
            bottleneck_output = self.bn_function(prev_features) # 做bn_func nrc1

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output))) # nrc2
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2 # override nn.ModuleDict

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers): # 增加i个_DenseLayer
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    # 张量拼接，实现densenet网络稠密连接
    def forward(self, init_features: Tensor) -> Tensor: # override nn.ModuleDict
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        # DenseNet使用过渡层来减半高和宽


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # Sequential: A sequential container. Modules will be added to it in the order they are passed in the constructor.
        # Alternatively, an ordered dict of modules can also be passed in.
        # OrderedDict: 神经网络模块为元素的有序字典
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config): # enumerate枚举, num_layers个densenetblock加1个transition
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features, # 通道数
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)  # //表示整数除法
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2 # ResNet里通过步幅为2的残差块在每个模块之间减小高和宽。DenseNet则使用过渡层来减半高和宽，并减半通道数

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight) # normal distribution, Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) # 用值val填充向量
                nn.init.constant_(m.bias, 0) # 用值val填充向量
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0) # 用值val填充向量

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1) # 第1个维度开始以后全推平ex:（2，3，4）to(2,12)
        out = self.classifier(out)
        return out

# 预训练网络的时候加载
def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    # 将正则表达式编译成 Pattern 对象
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs) # growth_rate, block_config, num_init_features