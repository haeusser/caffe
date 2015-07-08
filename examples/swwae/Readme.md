# Stacked What-Where Autoencoder for MNIST challenge

This model derives a stacked what-where autoencoder (SWWAE) from LeNet to achieve digit recognition.

## Architecture
   
      (softmax) # 
      ( loss  ) # C
         ^      # L
         |      # A
     +--------+ # S
     |  ip2   | # S
     +--------+ # I
         ^      # F
         |      # I
     +--------+ # E
     |  ip1   | # R
     +--------+ #
         ^      #
         |
     +--------+       +---------+     #
     |  pool2 |------>| unpool1 |     # 
     +--------+       +---------+     #
         ^                 |          # 
         |                 v          # 
     +--------+       +---------+     # 
     |  conv2 |       | deconv1 |     # 
     +--------+       +---------+     # 
         ^                 |          # S 
         |------(L2M)------|          # W
         |                 v          # W
     +--------+       +---------+     # A
     |  pool1 |       | unpool2 |     # E
     +--------+       +---------+     # 
         ^                 |          #
         |                 v          #
     +--------+       +---------+     # 
     |  conv1 |       | deconv2 |     # 
     +--------+       +---------+     # 
         ^                 |          # 
         |                 v          #
     /--------\ (Lrec) /--------\     # 
     |  data  |<------>| recons |     # 
     \--------/        \--------/     # 
   
     ###########      #############
       CONVNET          DECONVNET

The original model (LeNet) is the combination of the CONVNET part and the classifier (the inner product layers).

The stacked what-where autoencoder is the combination of the CONVNET and the DECONVNET parts.

The total loss is the weighted sum of the different losses :

* Lrec, which is the reconstruction loss between the intput image and the output of the DECONVNET.
* L2M, which is the reconstruction loss between the intermediate representation of the image and acts as a regularization loss
* The softmax loss, which is the loss produced by LeNet and specific to the task at hand
