# unionnet
Replementation of unionnet "Deep Learning from Multiple Noisy Annotators as A Union"

## Things to do
TODO List:
- [ ] Test the LabelMe dataset referring to the method of Max-Mix.
- [ ] Reimplement the dataset. dataloader. 
- [ ] Change the dataset to Chaoyang dataset.
- [ ] Add the compared methods, like Majority Vote Method, and comparing E2E method & MV method.
- [ ] Add the EM method.


## The setting of environment.
`
conda create -n union python=3.8 pip   # please do not install python=3.9 as the potential conflicts.
conda activate union  
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch   
pip install -r requirements.txt  
pip install ipdb  
pip install timm==0.5.4 # please make sure the version of timm matches pytorch's one.
pip install wandb
pip install protobuf==3.16.0  
`


## Running the code.
`bash run.sh # after modify the commands in the file.  `  
