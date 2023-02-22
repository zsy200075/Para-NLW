# conda info --envs
conda create -n mlm python=3.6 -y
conda activate mlm 

pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip uninstall transformers -y
# pip install transformers==4.9.2
pip install transformers==4.9.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install omegaconf -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install hydra -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install hydra.core -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install bitarray -i https://pypi.tuna.tsinghua.edu.cn/simple/ #2.3.7 
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install subword-nmt -i https://pypi.tuna.tsinghua.edu.cn/simple/






