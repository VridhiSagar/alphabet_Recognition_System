# alphabet_Recognition_System
Sign language is widely used by individuals with hearing impairment to communicate with each other conveniently using hand gestures. However, non-sign-language speakers find it very difficult to communicate with those with speech or hearing impairment since it interpreters are not readily available at all times. Many countries have their own sign language, such as American Sign Language (ASL) which is mainly used in the United States and the English-speaking part of Canada. The proposed system helps non-sign-language speakers in recognizing gestures used in American Sign Language. 

In this project, we are using SURF (Speeded up Robust Feature) on different algorithms like SVM and Naive Bayes for ASL gesture recognition and comparing the efficiency of  these models.

![](Picture1.png)

In this approach, firstly, the signs are captured using a webcam. First the input image is processed and skin masking is done. Then edge detection is used to detect the edge of the hand. After that SURF feature detection is used and then image is classified using SVM and Naive Bayes algorithm and the accuracy of both the algorithms is calculated. We have then applied a deep learning model-CNN and used it to predict the input gesture and the accuracy.
