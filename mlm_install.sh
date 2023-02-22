# conda info --envs
conda create -n mlm python=3.6 -y
conda activate mlm 

pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip uninstall transformers -y
# pip install transformers==4.9.2
pip install transformers==4.9.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/


# 安装pytorch
# cd ..
cd manuall_install_python_package
pip install torch-1.6.0+cu101-cp36-cp36m-linux_x86_64.whl

cd ..
cd ParaLS/ParaLS

pip install omegaconf -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install hydra -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install hydra.core -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install bitarray -i https://pypi.tuna.tsinghua.edu.cn/simple/ #2.3.7 
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install subword-nmt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# # 安装pip
# cd manuall_install_python_package/pip-19.0.1
# python setup.py install
# cd ..
# cd ..


# 安装datasets 压缩文件在MLM文件夹中
# yum install unzip zip
# unzip -o datasets-master.zip -d 文件夹
# 删除文件
# rm -rf datasets
# 修改文件名
# mv datasets-master datasets

# linux修改文件命令
# vim 文件名
# 找到编辑位置按“i”开始编辑
# 4、修改文件内容后退出：按ESC键
# 5、保存修改：
# （1）shift+“：”，使文件变成可查询状态
# （2）输入 wq！
# 6、不保存修改：
# （1）shift+“：”，使文件变成可查询状态
# （2）输入 q！

# 删除文件
# rm -rf 
# 查看隐藏文件
# ls -al


# "/home/user/.cache/huggingface/modules/datasets_modules/metrics/bertscore/23c058b03785b916e9331e97245dd43a377e84fb477ebdb444aff40629e99732/bertscore.py"