apt update && apt install zsh  vim -y && \
git clone https://mirrors.tuna.tsinghua.edu.cn/git/ohmyzsh.git && \
cd ohmyzsh/tools && \
yes y | REMOTE=https://mirrors.tuna.tsinghua.edu.cn/git/ohmyzsh.git sh install.sh && \
git clone https://githubfast.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
git clone https://githubfast.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' $HOME/.zshrc && \
sed -i 's/ZSH_THEME=\"robbyrussell\"/ZSH_THEME=\"fino\"/' $HOME/.zshrc && \
source ~/.zshrc 


cd / && rm -rf ohmyzsh && \
conda init zsh && \
rm ~/.condarc && \
echo "channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc  && \
yes y | conda clean -i && \
source ~/.zshrc && \
conda create -n yi python=3.8 -y && \
conda activate yi && \
python -m pip install --upgrade pip && \
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
cd code && \
git clone https://githubfast.com/starmountain1997/Yi.git && \
cd Yi && \
git checkout npu && \
pip install -r requirements.txt

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
git clone https://github.com/starmountain1997/Yi.git && \
cd Yi && \
git checkout npu && \
pip install -r requirements.txt && \
pip install git+https://github.com/starmountain1997/AMP.git && \
./finetune/scripts/run_sft_Yi_6b_zsh.sh