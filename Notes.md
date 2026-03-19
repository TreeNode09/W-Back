- LPN = **L**earning **P**arity with **N**oise
- LDPC = **L**ow-**D**ensity **P**arity-**C**heck

#### Use Python 3.11 venv on Remote Computer

```
sudo apt update
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

git clone https://github.com/XuandongZhao/PRC-Watermark
cd PRC

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```