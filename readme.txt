使用方法：
打开命令行后
1.安装依赖包
pip install -r requirements.txt
2.训练模型
python CNN_Train.py
3.用得到的最好模型预测
python CNN_Test.py

[注]当前默认参数为：
BATCH_SIZE=64
EPOCHS=30
MOMENTUM=0.5
LEARNING_RATE=0.01
optimizer=Adamax
