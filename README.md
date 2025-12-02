# Python TsFile测试工具说明文档

----

# 环境

- Python > 3.8

# 依赖

步骤一：编译源码生成wheel包

```bash
git clone https://github.com/apache/tsfile.git
cd tsfile
mvn clean install -P with-python -DskipTests # wheel包位于tsfile根目录下/python/dist中

```

步骤二：引用wheel包

```bash
git clone https://github.com/thulab/python-tsfile-api-test.git
cd python-tsfile-api-test
pip3 install tsfile-*.whl
# 查看包的详细信息：pip3 show tsfile
# 删除旧引用：pip3 uninstall tsfile
```

# 测试

## 自动化测试——pytest

### 安装

```
# 1、安装（Linux或Windows）
pip3 install pytest

# 2、检测
pytest --version
```

### 使用

```bash
cd tests
pytest
```

### 拓展

- 生成JSON文件报告

```Bash
# 安装插件
pip install pytest-json
# 使用（生成的文件位于当前执行目录下）
pytest --json=report.json
```

- 生成HTML文件报告

```Bash
# 安装插件
pip install pytest-html
# 使用（生成的文件位于当前执行目录下）
pytest --html=report.html
```

## 覆盖率测试——pytest-cov

### 安装

```
# 安装代码覆盖模块（确保先安装好自动化测试工具——pytest）
pip3 install pytest-cov
```

### 使用

```bash
# 1、收集要测量覆盖率的源码放到程序根目录的tsfile下

# 2、执行下面命令进行代码覆盖率测试并生成报告文件（已开启分支覆盖）
cd tests
pytest --cov=tsfile --cov-report=html --cov-branch
```

生成的报告默认位于 `${python-client-api-test}/test/htmlcov/ ` 下的`index.html` 文件
生成json覆盖率报告：`pytest --cov=tsfile --cov-report=json --cov-config=.coveragerc --cov-branch`

## 性能测试——pytest-benchmark

暂未实现

## 综合运行

```bash
# 覆盖率+自动化
pytest --cov=tsfile --cov-report=html --cov-branch --html=report.html
```
