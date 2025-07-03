<p align="center">
  <img src="docs/icon_dice.png" alt="LotteryCodec Icon" width="100"/>
</p>

<h1 align="center">
 LotteryCodec: Searching the Implicit Representation in a Random Network for Low-Complexity Image Compression
</h1>

<p align="center">
  <a href="https://eedavidwu.github.io/">Haotian Wu</a>&nbsp;&nbsp;
  <a href="https://gp-chen.github.io/">Gongpu Chen</a>&nbsp;&nbsp;
  <a href="https://www.commsp.ee.ic.ac.uk/~pld/">Pier Luigi Dragotti</a>&nbsp;&nbsp;
  <a href="https://www.imperial.ac.uk/information-processing-and-communications-lab/people/deniz/">Deniz Gündüz</a>  
  <br/>
  <strong>Imperial College London</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2507.01204" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv">
  </a>
  <a href="https://eedavidwu.github.io/LotteryCodec/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
</p>

## 📣 Latest Updates
-  **[2025-06-28]** 📝 *More results with various mask ratio selections are updated in [results](https://github.com/eedavidwu/LotteryCodec/tree/master/results), which is also provided in the updated [paper](https://arxiv.org/abs/2507.01204v1). (Training CLIC2020 with more ratios costs lots of resources for us :) .)*
-  **[2025-06-22]** 📝 *Resources, such as baseline implementations (VTM-19.1) with its datapoints are now updated on [resources](https://github.com/eedavidwu/LotteryCodec/blob/master/resource/README.md).*
- **[2025-06-21]** 📝 *Detailed intermediate results are now released on [results](https://github.com/eedavidwu/LotteryCodec/tree/master/results).*
- **[2025-06-01]** 🎉 *LotteryCodec has been accepted to **ICML 2025** as a **Spotlight**!*

## 🔑 Key Takeaways

- **LotteryCodec** introduces a novel overfitted codec for low-complexity image compression. Instead of training a synthesis neural function, LotteryCodec searches for a well-performing subnetwork within a randomly initialized network!

- A **Lottery Codec Hypothesis** is introduced: Win a lottery ticket as your own image codec!


<p align="center">
  <img src="docs/LCH.png" width="1000"/>
</p>

- To simplify the searching process and improve the performance, **LotteryCodec** employs a modulation-based new paradigm.

![sicl](docs/SuperMask_fig_1.png)

<p align="center">
  <img src="docs/Fig_3_masked_sys.png" width="1000"/>
</p>

## About this code
The LotteryCodec codebase is written in Python and provides fast configurations for the training. The core module structure is as follows:
```
LotteryCodec/
├── dataset/                          # Folder for dataset.
│   ├── CLIC2020.                
│   ├── Kodak/                   
├── enc/                          # Folder for encoding functions
│   ├── training/                
│   ├── utils/                 
├── models/                       # Main model.
│   └── model.py                  
├── resource/                       # Resources for the baselines.
├── results            # Experimental results for various models.
├── utils            # Code for sub-model and functions such as quantization/ARM/...
├── train.py

```
The code is heavily based on the Cool-Chic project, an outstanding open-source work :). For additional resources and attribution (such as engineering optimization), please refer to their project page:  <a href="https://github.com/Orange-OpenSource/Cool-Chic">Cool-Chic</a>  

## Results:
A better RD performance:
<p align="center">
  <img src="docs/RD_Kodak_CLIC.png" width="1000"/>
</p>

Towards BD-rate vs. flexible complexity (a) Kodak and (b).CLIC2020:
<p align="center">
  <img src="docs/BD-rate_com.png" width="1000"/>
</p>

## Contact
- Haotian Wu: haotian.wu17@imperial.ac.uk

Please open an issue or submit a pull request for issues, or contributions.

## 💼 License

<a href="https://opensource.org/licenses/MIT" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
</a>

## Citation

If you find our resource/idea is helpful, please cite our paper:

```
  @article{LotteryCodec,
    title={LotteryCodec: Searching the Implicit Representation in a Random Network for Low-Complexity Image Compression},
    author={Haotian Wu, Gongpu Chen, Pier Luigi Dragotti, and Deniz Gündüz},
    journal={International Conference on Machine Learning (ICML) 2025},
    year={2025}
  }

```

