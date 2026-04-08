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

- Python > 3.8 （推荐3.10 到 3.12）
- C++ 环境 （编译依赖需要）

# 依赖

步骤一：编译源码生成wheel包

```bash
git clone https://github.com/apache/tsfile.git
cd tsfile
mvn clean install -P with-python -DskipTests
```

wheel包位于tsfile根目录下/python/dist中

常见问题

```bash
# 1、若出现缺少clang-format错误，请执行如下命令安装clang-format
# Ubuntu
sudo apt install clang-format-17
# Windows（在管理员权限的PowerShell中执行）若没有安装choco，执行：Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install llvm --version 17.0.6 -y
# 验证（Windows验证需要重新打开一个窗口）
clang-format --version

# 2、若出现缺少MinGW错误，请执行如下命令安装MinGW
# Windows（在管理员权限的PowerShell中执行）
choco install mingw -y
# 验证（Windows验证需要重新打开一个窗口）
gcc --version
g++ --version

# 3、若出现clang编译器找不到Windows SDK库文件错误（如kernel32.lib等）
# 原因：LLVM的clang需要Windows SDK，而MinGW Makefiles配置应使用gcc/g++
# 解决方案一：临时设置环境变量（cmd中执行）
set CC=gcc
set CXX=g++
rd /s /q cpp\target\build  # 清理构建缓存
mvn clean install -P with-python -DskipTests

# 解决方案二：永久设置环境变量（避免每次编译都要设置）
# Windows系统环境变量设置方法：
# （1）调整PATH顺序：将 C:\ProgramData\mingw64\mingw64\bin 移到 C:\Program Files\LLVM\bin 之前
#     打开"系统属性" → "高级" → "环境变量" → 编辑系统变量"Path" → 移动MinGW路径到LLVM上方
# （2）或在系统环境变量中添加：
#     新建变量 CC=gcc，CXX=g++
# 命令行快速设置（管理员权限cmd）：
setx CC "gcc" /M
setx CXX "g++" /M
```

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
pip3 install pytest-json
# 使用（生成的文件位于当前执行目录下）
pytest --json=report.json
```

- 生成HTML文件报告

```Bash
# 安装插件
pip3 install pytest-html
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
