# LSNet

The code will be available after our paper is accepted
——————————————————————————————————————————————————————————————————————
2024.5.28 
Since we uploaded the paper to Arxiv, we decided to release the code. The following is an introduction.


Note!!!

Our model has the following limitations (as shown in the paper):

Some results will have discontinuous artifacts in small rectangular areas. Changing the parameter of the second convolution kernel of self.conv4 and self.conv3 to 3 will alleviate this problem.


If you find this code useful or use our model, please cite

@article{zhou20247k,
  title={A 7K Parameter Model for Underwater Image Enhancement based on Transmission Map Prior},
  author={Zhou, Fuheng and Wei, Dikai and Fan, Ye and Huang, Yulong and Zhang, Yonggang},
  journal={arXiv preprint arXiv:2405.16197},
  year={2024}
}
