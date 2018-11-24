# H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes, TMI 2018. 
by [Xiaomeng Li](https://scholar.google.com/citations?user=uVTzPpoAAAAJ&hl=en), [Hao Chen](http://appsrv.cse.cuhk.edu.hk/~hchen/), [Xiaojuan Qi](https://xjqi.github.io/), [Qi Dou](http://appsrv.cse.cuhk.edu.hk/~qdou/), [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

This repository is for our TMI 2018 paper '[H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes](http://arxiv.org/pdf/1709.07330.pdf)'.


### Usage


1. Data preprocessing: 
   Download dataset from: [Liver Tumor Segmentation Challenge](https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE).   
   Then put 131 training data with segmentation masks under "data/TrainingData/" and 70 test data under "data/TestData/".  
   Run:
   ```shell 
   python preprocessing.py 
   ```


2. Test our model:
   Download liver mask from [LiverMask](https://drive.google.com/file/d/14DFUoX877dWjaiau_5c5aI3j4JYxdEMm/view?usp=sharing) and put them in folder: 'livermask'.   
   Download model from [Model](https://drive.google.com/file/d/1sZgqnqjF_M3Bv_9M2kkfPZyPTrIoVttB/view?usp=sharing) and put them in folder: 'model'.   
   run:
   ```shell
   python test.py
   ```

3. Train 2D DenseUnet:
    First, you need to download the pretrained model from [ImageNet Pretrained](https://drive.google.com/file/d/195LqUBwueXqhmRcznK20S7AK6NtGEHBM/view?usp=sharing), extract it and put it in folder 'model'.
    Then run:
   ```shell
   sh bash_train.sh
   ```

4, Train H-DenseUnet:

	First, Load your trained 2d model. 
	Then run: 
	```shell 
	sh bash_trainhybrid.sh
	```

## Citation

If H-DenseUNet is useful for your research, please consider citing:

	@article{li2018h,
	  title={H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes},
	  author={Li, Xiaomeng and Chen, Hao and Qi, Xiaojuan and Dou, Qi and Fu, Chi-Wing and Heng, Pheng-Ann},
	  journal={IEEE Transactions on Medical Imaging},
	  year={2018},
	  publisher={IEEE}
	}

### Questions

Please contact 'xmli@cse.cuhk.edu.hk'

