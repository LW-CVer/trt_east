# trt_east


支持动态输入 <br>

训练onnx的代码 <br>

onnx文件地址: https://pan.baidu.com/s/11GDJB0fqTIsgpK8P2a4sJw 
提取码: plc3 <br>

使用的icdar2015数据集训练的模型(不太确定了) <br>

trt 7.1.3.4 <br>
cuda 10.2 <br>
cudnn 8.0 <br>
gcc 7.3 <br>

mkdir build && cd build
cmake .. <br>
make <br>
./test/test_trt_east <br>

