# LSNet

The code will be available after our paper is accepted<br>
——————————————————————————————————————————————————————————————————————<br>
2024.5.28 <br>
Since we uploaded the paper to Arxiv, we decided to release the code. The following is an introduction.<br>


Note!!!<br>

Our model has the following limitations (as shown in the paper):<br>

Some results will have discontinuous artifacts in small rectangular areas. Changing the parameter of the second convolution kernel of self.conv4 and self.conv3 to 3 will alleviate this problem.<br>








If you find this code useful or use our model, please cite<br>

@article{zhou20247k,<br>
  title={A 7K Parameter Model for Underwater Image Enhancement based on Transmission Map Prior},<br>
  author={Zhou, Fuheng and Wei, Dikai and Fan, Ye and Huang, Yulong and Zhang, Yonggang},<br>
  journal={arXiv preprint arXiv:2405.16197},<br>
  year={2024}<br>
}
