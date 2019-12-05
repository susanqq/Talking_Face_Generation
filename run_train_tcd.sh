
train_cnn(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --save_dir 'save_CAAE_TCD_CAE/'$3 \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_TCD' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/data/TCD/merge_train_list.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}


train_cnn_VideoAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --num_frames_D 20 \
  --discriminator_v 'video_3D' \
  --ckpt 'save_CAAE_TCD_CAE/face_unet_reduce_128/model_final.pt' \
  --save_dir 'save_CAAE_TCD_CAE/'$3 \
  --use_seq true \
  --test_dir  '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_TCD' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/data/TCD/merge_train_list.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}


train_cnn_FrameVideoAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --num_frames_D 20 \
  --discriminator 'frame' \
  --discriminator_v 'video_3D' \
  --ckpt 'save_CAAE_TCD_CAE/face_unet_reduce_128/model_final.pt' \
  --save_dir 'save_CAAE_TCD_CAE/'$3 \
  --use_seq true \
  --test_dir  '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_TCD' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/data/TCD/merge_train_list.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}



train_rnn(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --rnn_type 'GRU' \
  --ckpt 'save_CAAE_TCD_CAE/face_unet_reduce_128/model_final.pt' \
  --save_dir 'save_CAAE_TCD_RAE_ia/'$3 \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_TCD' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/data/TCD/merge_train_list.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}



train_rnn_FrameVideoAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --rnn_type 'GRU' \
  --num_frames_D 20 \
  --discriminator 'frame' \
  --discriminator_v 'video_3D' \
  --ckpt 'save_CAAE_TCD_RAE_ia/GRU_unet_128/model_final.pt' \
  --save_dir 'save_CAAE_TCD_RAE_ia/'$3 \
  --use_seq true \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_TCD' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/data/TCD/merge_train_list.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}

#
#train_rnn_FrameVideoAdv 6,7 4 'GRU_unet_128_FrameVideoAdv'

train_cnn_VideoAdv 13 4 'face_unet_reduce_128_videoAdv'

#train_cnn_FrameVideoAdv 14 4 'face_unet_reduce_128_FrameVideoAdv'