# training
python train.py --dataroot=./arctic_dataset --resize_or_crop none --netG downandupsample --display_id 0 


# testing
# A -> B
python test.py --dataroot=./arctic_dataset --resize_or_crop none --netG downandupsample --phase val --model_suffix _A --direction AtoB

# B -> A
python test.py --dataroot=./arctic_dataset --resize_or_crop none --netG downandupsample --phase val --model_suffix _B --direction BtoA
