# 1. Hardware specification<br>
Here is the hardware we used to produce the result

* GPU specs: NVIDIA Geforce RTX A5000 GPU 24GB<br>
* Number of GPUs: 1 <br>
* Memory: 1TB<br>

# 2. OS and softwares<br>
* OS: Ubuntu 20.04 LTS<br>
* Cuda: 12.2<br> 
* Python: 3.8.18<br>
* Pytorch: 1.11.0

# 3. Dataset download<br> 
The dataset can be accessed from the following link：<br>
* rimone.isaatc.ull.es<br>
* https://refuge.grand-challenge.org/<br>
* http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php
  
# 4. Main contributions<br> 

This paper introduces a novel edge enhancement network named EE-TransUNet for optic cup and optic disc segmentation. It builds upon the TransUNet network, enhancing feature abstraction and image perception using cascaded convolutional fusion and channel-shuffling of multiple expansion fusion blocks. This improvement aims to enhance segmentation accuracy, especially at the edges of optic cup and optic disc. Ablation and comparison experiments have been conducted on three public datasets of optic disc optic cups respectively, and the experiments demonstrate the effectiveness of the modules proposed in this paper as well as the superior performance of the proposed network over the current state-of-the-art networks. The main contributions of this paper are as follows:<br>
* A novel edge-enhanced segmentation network, named EE-TransUNet, proposed for optic cup and optic disc segmentation tasks. This network is distinguished by a series of techniques aimed at enhancing feature abstraction expression and network perception. It demonstrates excellent performance in optic cup and optic disc segmentation tasks, particularly in fine segmentation at their edges.<br>
* To solve the problem of insufficient and not rich enough feature expression in the original TransUNet, this paper proposes the Cascaded Convolutional Fusion (CCF) block. By adding the CCF block, the abstract expression of features can be enhanced and the information of the original features is retained, thus improving the nonlinear fitting ability. In this way, the model is able to better capture the subtle features in the image.<br>
* In the original TransUNet model, the skip connection suffers from insufficient or loss of information transfer. To overcome this problem, this paper proposes the Channel Shuffling Multiple Expansion Fusion block (CSMF). The incorporation of the CSMF block can improve the network's capacity to perceive and characterize image features, particularly at the skip connections. Trough the employment of multiple expansion fusions, the CSMF block effectively enhances the network's segmentation accuracy at the edges. This aids in capturing the edge information of the target structure more accurately, thereby improving the overall segmentation accuracy.<br>


# 5. About weights<br> 

* If you want to reproduce the results from the paper, you can click on the link below to download our weights<br>

Link：https://pan.baidu.com/s/1IQsiUMnmRIYzJ2qI1o0OPQ?pwd=gwcy 


# 6. Future issues<br> 
If you find any problems running the code, or have any questions regarding the solution, please contact me at: wangyunyu716@gmail.com and create an issue on the Repo's Issue tab.
