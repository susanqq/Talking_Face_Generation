
train_cnn(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'hk' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --save_dir 'save_CAAE_VOX_CAE/'$3 \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/DATA/filelist_shuffle_vox_filterAll_warp_input_more.txt' \
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
  --ckpt 'save_CAAE_VOX_CAE/face_unet_reduce_128_VideoAdv/model_G11.pt' \
  --save_dir 'save_CAAE_VOX_CAE/'$3 \
  --use_seq true \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/DATA/filelist_shuffle_vox_filterAll_warp_input_more.txt' \
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
  --ckpt 'save_CAAE_VOX_CAE/face_unet_reduce_128/model_final.pt' \
  --save_dir 'save_CAAE_VOX_CAE/'$3 \
  --use_seq true \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/DATA/filelist_shuffle_vox_filterAll_warp_input_more.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}

train_cnn_lipAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --num_seq_length 40 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --use_lip true \
  --use_word_label true \
  --use_seq true \
  --discriminator_lip 'lip_read' \
  --save_dir 'save_CAAE_VOX_CAE/'$3 \
  --ckpt 'save_CAAE_VOX_CAE/face_unet_reduce_128/model_final.pt' \
  --ckpt_lipmodel '../lip_read_pytorch/lip_read_model/2LSTM_seq_350ms_non_align_rgb/model_final.pt' \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list 'lip_read_training_list/LRW_VOX_training_list_merge.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}


train_cnn_ThreeAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --num_seq_length 40 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --use_lip true \
  --use_word_label true \
  --use_seq true \
  --num_frames_D 11 \
  --discriminator 'frame' \
  --discriminator_v 'video_3D' \
  --discriminator_lip 'lip_read' \
  --save_dir 'save_CAAE_VOX_CAE/'$3 \
  --ckpt 'save_CAAE_VOX_CAE/face_unet_reduce_128_FrameVideoAdv/model_G21.pt' \
  --ckpt_lipmodel '../lip_read_pytorch/lip_read_model/2LSTM_seq_350ms_non_align_rgb/model_final.pt' \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list 'lip_read_training_list/LRW_VOX_training_list_merge.txt' \
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
  --ckpt 'save_CAAE_LRW_CAE/face_unet_reduce_128_node6/model_6.pt' \
  --save_dir 'save_CAAE_VOX_RAE_ia/'$3 \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/DATA/filelist_shuffle_vox_filterAll_warp_input_more.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}


train_rnn_FrameAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --rnn_type 'GRU' \
  --discriminator 'frame' \
  --ckpt 'save_CAAE_VOX_RAE_ia/GRU_unet_128/model_final.pt' \
  --save_dir 'save_CAAE_VOX_RAE_ia/'$3 \
  --use_seq true \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/DATA/filelist_shuffle_vox_filterAll_warp_input_more.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}



train_rnn_VideoAdv(){
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
  --discriminator_v 'video_3D' \
  --ckpt 'save_CAAE_VOX_RAE_ia/GRU_unet_128/model_final.pt' \
  --save_dir 'save_CAAE_VOX_RAE_ia/'$3 \
  --use_seq true \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/DATA/filelist_shuffle_vox_filterAll_warp_input_more.txt' \
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
  --ckpt 'save_CAAE_VOX_RAE_ia/GRU_unet_128/model_final.pt' \
  --save_dir 'save_CAAE_VOX_RAE_ia/'$3 \
  --use_seq true \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list '/SLP_Extended/susan.s/Documents/Speech2Vid/DATA/filelist_shuffle_vox_filterAll_warp_input_more.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}



train_rnn_ThreeAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --num_seq_length 40 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --rnn_type 'GRU' \
  --use_lip true \
  --use_word_label true \
  --use_seq true \
  --num_frames_D 11 \
  --discriminator 'frame' \
  --discriminator_v 'video_3D' \
  --discriminator_lip 'lip_read' \
  --save_dir 'save_CAAE_VOX_RAE_ia/'$3 \
  --ckpt 'save_CAAE_VOX_RAE_ia/face_unet_reduce_128_FrameVideoAdv/model_G32.pt' \
  --ckpt_lipmodel '../lip_read_pytorch/lip_read_model/2LSTM_seq_350ms_non_align_rgb/model_final.pt' \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --filename_list 'lip_read_training_list/LRW_VOX_training_list_merge.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}
#train_cnn_lipAdv 4,5 24 'face_unet_reduce_128_lipAdv'

#train_cnn_VideoAdv 8,9 4 'face_unet_reduce_128_VideoAdv'

train_cnn_ThreeAdv 6 16 'face_unet_reduce_128_ThreeAdv'

#train_rnn_ThreeAdv 6,7 20 'face_unet_reduce_128_ThreeAdv'
