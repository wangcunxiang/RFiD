<div align="center">
  <h1> Rational-Fusion-in-Decoder </h1>
  
  [![License: Apache-2.0](https://img.shields.io/crates/l/Ap?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
</div>

  # 📌 Table of Contents
- [Introduction](#-introduction)
- [Development](#-development)
- [Data](#-data)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)
  
# 🚀 Introduction
This repository is for the ACL-2023 findings paper "RFiD: Towards Rational Fusion-in-Decoder for Open-Domain Question Answering" [paper](https://arxiv.org/abs/2305.17041).

# 💻 Development
## Setup
Clone the repository from GitHub and install:
```
git clone https://github.com/wangcunxiang/RFiD.git
cd RFiD/
pip install -e ./
```
## Train Script
```
name={name}
CUDA_VISIBLE_DEVICES=X python train_reader.py \
        --train_data path/to/train_file.json \
        --eval_data path/to/dev_file.json \
        --model_size [large]/[base] \
        --per_gpu_train_batch_size 1 \
        --per_gpu_eval_batch_size 1 \
        --accumulation_steps 64 \
        --total_steps 320000 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --n_context 100 \
        --add_loss binary \
        --cat_emb \
        --name ${name} \
        --checkpoint_dir checkpoint

```
## Test Script
```
name={name}
CUDA_VISIBLE_DEVICES=X python test_reader.py \
        --eval_data path/to/test_file.json \
        --model_path path/to/checkpoint_dir/ \
        --per_gpu_eval_batch_size 1 \
        --n_context 100 \
        --write_results \
        --answer_maxlength 30 \
        --name ${name} \
        --add_loss binary \
        --cat_emb \
        --output_attentions \
        --sum_golden_cross_att \
        --checkpoint_dir checkpoint
```
If your want to do cross attention alaysis:  
```
name={name}
CUDA_VISIBLE_DEVICES=X python test_reader.py \
        --eval_data path/to/test_file.json \
        --model_path path/to/checkpoint_dir/ \
        --per_gpu_eval_batch_size 1 \
        --n_context 100 \
        --write_results \
        --answer_maxlength 30 \
        --name ${name} \
        --add_loss binary \
        --cat_emb \
        --output_attentions \
        --sum_golden_cross_att \
        --checkpoint_dir checkpoint
```

[//]: # (## Checkpoints)

[//]: # (We release our best-performed checkpoints on Natural Questions and TriviaQA publically.)


# 📝 Data
### Following [FiD](https://github.com/facebookresearch/FiD), we use the same data and format.
## Download data
NaturalQuestions and TriviaQA data can be downloaded using get-data.sh. Both datasets are obtained from the original source and the wikipedia dump is downloaded from the DPR repository. In addition to the question and answers, this script retrieves the Wikipedia passages used to trained the released pretrained models.

## Data format
The expected data format is a list of entry examples, where each entry example is a dictionary containing
```
id: example id, optional
question: question text
target: answer used for model training, if not given, the target is randomly sampled from the 'answers' list
answers: list of answer text for evaluation, also used for training if target is not given
ctxs: a list of passages where each item is a dictionary containing - title: article title - text: passage text
```
Entry example:
```
{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}
```
# 📜 License

This repository is released under the [Apache-2.0 License](LICENSE).

# 📚 Citation

If you find this repository useful, please cite it as follows:
```bibtex
@misc{wang2023rfid,
      title={RFiD: Towards Rational Fusion-in-Decoder for Open-Domain Question Answering}, 
      author={Cunxiang Wang and Haofei Yu and Yue Zhang},
      year={2023},
      eprint={2305.17041},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## 📮 Contact
If you have any questions or feedback, please feel free to reach out at wangcunxiang@westlake.edu.cn.
