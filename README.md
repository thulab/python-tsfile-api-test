# Python TsFile测试工具使用说明文档

----

测试工具部署：[‌‍﻿⁠‌‍⁠‍‌‬﻿‬﻿‍‌‬‬⁠‌‬‬⁠‍‌‬‬⁠﻿‬‬⁠TsFile 接口测试工具部署和使用 - 飞书云文档](https://timechor.feishu.cn/docx/Xw64d5FFZoQLKQxZW2UcwGHUnwb)

## 自动化测试工具——pytest

```bash
cd tests
pytest
```

## 代码覆盖率工具——pytest-cov

```bash
# 1、收集要测量覆盖率的源码目录（tsfile）

# 2、复制上面 .coveragerc 文件放到测试目录下，执行下面命令进行代码覆盖率测试并生成报告文件（已开启分支覆盖）
cd tests
pytest --cov=tsfile --cov-report=html --cov-branch
```

生成的报告默认位于 `${python-client-test}/test/htmlcov/ ` 下的`index.html` 文件
生成json覆盖率报告：`pytest --cov=tsfile --cov-report=json --cov-config=.coveragerc --cov-branch`

## 性能测试工具——pytest-benchmark

暂未实现

## 综合运行

```bash
# 1、覆盖率+自动化
pytest --cov=tsfile --cov-report=html --cov-branch --html=report.html
```
