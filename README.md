# Python TsFile测试工具说明文档

当前测试工具主要用于测试TsFile的Python API。

----

# 项目结构

```
python-tsfile-api-test/
├── data/                          # 测试数据目录
│   ├── csv/                       # CSV格式测试数据
│   └── tsfile/                    # 存放测试生成TsFile测试数据
├── example/                       # 示例代码目录
├── tests/                         # 测试代码目录
│   ├── table/                     # 表结构相关测试
│   ├── tree/                      # 树结构相关测试
│   └── coveragerc                 # 覆盖率配置文件
├── tsfile/                        # 存放TsFile源码目录(用于覆盖率测试)
├── .gitignore                     # Git忽略配置
└── README.md                      # 项目说明文档
```

# 环境

- Python > 3.8

# 依赖

步骤一：编译源码生成wheel包

```bash
git clone https://github.com/apache/tsfile.git
cd tsfile
mvn clean install -P with-python -DskipTests 
```

wheel包位于tsfile根目录下/python/dist中

步骤二：引用wheel包

```bash
git clone https://github.com/thulab/python-tsfile-api-test.git
cd python-tsfile-api-test
# 引入wheel包
pip3 install /path/to/tsfile-*.whl
# 引入其他需要的
pip3 install numpy
pip3 install pandas
pip3 install pyarrow
# 查看包的详细信息：pip3 show tsfile
# 删除旧引用：pip3 uninstall tsfile
```

# 测试

## 功能测试——pytest

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
# 1、收集要测量覆盖率的源码放到程序根目录的tsfile下（注意：测试完成后请删除源码，以防功能测试引用的是源码而非whl包）

# 2、执行下面命令进行代码覆盖率测试并生成报告文件（已开启分支覆盖）
cd tests
pytest --cov=tsfile --cov-report=html --cov-branch
```

生成的报告默认位于项目根目录下的`tests/htmlcov/`目录中的`index.html`文件
生成json覆盖率报告：`pytest --cov=tsfile --cov-report=json --cov-config=.coveragerc --cov-branch`

## 性能测试——pytest-benchmark

暂未实现

## 综合运行

```bash
cd tests
# 覆盖率+自动化
pytest --cov=tsfile --cov-report=html --cov-branch --html=report.html
```
