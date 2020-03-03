# ocr_shixin

实现定长验证码的识别. 使用 `tensorflow` `2.1` 版的 `keras` 高级接口来训练模型.
模型使用了 `cnn`, `rnn`, `ctc` 来预测 `4` 位的验证码.

## 教程

[tutorial](tutorial.ipynb)

## tf-serving

[使用 REST 训练和提供模型](https://tensorflow.google.cn/tfx/tutorials/serving/rest_simple)

## docker 部署

```bash
docker run -p 8501:8501 --mount type=bind,source=d:/workspace/ocr_shixin/Shixin/,target=/models/Shixin -e MODEL_NAME=Shixin -t -d tensorflow/serving
```
