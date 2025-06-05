# Python TsFile测试工具使用说明文档

----

## 依赖引用

编译源码生成wheel包

```bash
git clone https://github.com/apache/tsfile.git
cd tsfile
mvn clean install -P with-python -DskipTests # wheel包位于tsfile根目录下/python/dist中

```

引用wheel包

```bash
git clone https://github.com/thulab/python-tsfile-test.git
cd python-tsfile-test
pip3 install tsfile-*.whl
# 删除旧引用：pip3 uninstall tsfile
```

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
