
train_cnn(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 50 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --use_npy true \
  --save_dir $3 \
  --test_dir 'testing/test_LRW' \
  --ckpt 'save_CAAE_LRW_CAE/face_unet_reduce_128_non_align_node7/model_1.pt' \
  --filename_list 'preprocess_LRW/train_list_LRW_non_align_subset2_npy.txt' \
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
  --num_frames_D 11 \
  --discriminator_v 'video_3D' \
  --save_dir $3 \
  --use_npy true \
  --test_dir 'testing/test_LRW' \
  --ckpt 'save_CAAE_LRW_CAE/face_unet_reduce_128_non_align_subset/model_G10.pt' \
  --filename_list 'preprocess_LRW/train_list_LRW_non_align_subset2_npy.txt' \
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
  --num_frames_D 11 \
  --discriminator 'frame' \
  --discriminator_v 'video_3D' \
  --save_dir $3 \
  --use_npy true \
  --test_dir 'testing/test_LRW' \
  --ckpt 'save_CAAE_LRW_CAE/face_unet_reduce_128_non_align_subset/model_G10.pt' \
  --filename_list 'preprocess_LRW/train_list_LRW_non_align_subset2_npy.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}

train_cnn_lipAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 20 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --save_dir $3 \
  --use_npy true \
  --use_lip true \
  --discriminator_lip 'lip_read' \
  --use_word_label true \
  --ckpt 'save_CAAE_LRW_CAE/face_unet_reduce_128_non_align_subset/model_G10.pt' \
  --ckpt_lipmodel '../lip_read_pytorch/lip_read_model/2LSTM_seq_350ms_non_align_rgb/model_final.pt' \
  --test_dir 'testing/test_LRW' \
  --filename_list 'lip_read_training_list/LRW_training_list_non_align_subset2_npy_128.txt' \
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
  --ckpt 'save_CAAE_LRW_CAE/face_unet_reduce_128_non_align_node7/model_1.pt' \
  --save_dir $3 \
  --use_npy true \
  --test_dir 'testing/test_LRW' \
  --filename_list 'preprocess_LRW/train_list_LRW_non_align_subset2_npy.txt' \
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
  --ckpt 'save_CAAE_LRW_RAE/GRU_unet_128_non_align/model_2.pt' \
  --save_dir $3 \
  --use_npy true \
  --test_dir 'testing/test_LRW' \
  --filename_list 'preprocess_LRW/train_list_LRW_non_align_subset2_npy.txt' \
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
  --num_frames_D 11 \
  --discriminator_v 'video_3D' \
  --ckpt 'save_CAAE_LRW_RAE/GRU_unet_128_non_align/model_2.pt' \
  --save_dir $3 \
  --use_npy true \
  --test_dir 'testing/test_LRW' \
  --filename_list 'preprocess_LRW/train_list_LRW_non_align_subset2_npy.txt' \
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
  --num_frames_D 11 \
  --discriminator 'frame' \
  --discriminator_v 'video_3D' \
  --ckpt 'save_CAAE_LRW_RAE/GRU_unet_128_non_align/model_2.pt' \
  --save_dir $3 \
  --use_npy true \
  --test_dir 'testing/test_LRW' \
  --filename_list 'preprocess_LRW/train_list_LRW_non_align_subset2_npy.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}

train_rnn_lipAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 20 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --rnn_type 'GRU' \
  --save_dir $3 \
  --use_npy true \
  --use_lip true \
  --discriminator_lip 'lip_read' \
  --use_word_label true \
  --ckpt 'save_CAAE_LRW_RAE/GRU_unet_128_non_align/model_4.pt' \
  --ckpt_lipmodel '../lip_read_pytorch/lip_read_model/2LSTM_seq_350ms_non_align_rgb/model_final.pt' \
  --test_dir 'testing/test_LRW' \
  --filename_list 'lip_read_training_list/LRW_training_list_non_align_subset2_npy_128.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}

train_rnn_ThreeAdv(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore train.py  \
  --epoch 20 \
  --batch_size $2 \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --rnn_type 'GRU' \
  --save_dir $3 \
  --use_npy true \
  --use_lip true \
  --num_frames_D 11 \
  --discriminator 'frame' \
  --discriminator_v 'video_3D' \
  --discriminator_lip 'lip_read' \
  --use_word_label true \
  --ckpt 'save_CAAE_LRW_RAE/GRU_unet_128_non_align_FrameVideoAdv/model_G4.pt' \
  --ckpt_lipmodel '../lip_read_pytorch/lip_read_model/2LSTM_seq_350ms_non_align_rgb/model_final.pt' \
  --test_dir 'testing/test_LRW' \
  --filename_list 'lip_read_training_list/LRW_training_list_non_align_subset2_npy_128.txt' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}

# train_cnn 2,3 48 'save_CAAE_LRW_CAE/face_unet_reduce_128_non_align_node7'
# train_cnn_lipAdv 7 40 'save_CAAE_LRW_CAE/face_unet_reduce_128_non_align_lipAdv'

# train_rnn_FrameAdv 0,1 48 'save_CAAE_LRW_RAE/GRU_unet_128_non_align_frameAdv'

# train_rnn_VideoAdv 6,7 32 'save_CAAE_LRW_RAE/GRU_unet_128_non_align_VideoAdv'

# train_rnn_FrameVideoAdv 4,5 32 'save_CAAE_LRW_RAE/GRU_unet_128_non_align_FrameVideoAdv'

train_rnn_lipAdv 2,3 40 'save_CAAE_LRW_RAE/GRU_unet_128_non_align_lipAdv'

#train_cnn_FrameVideoAdv 0,1 40 'save_CAAE_LRW_CAE/face_unet_reduce_128_non_align_FrameVideoAdv'

#train_rnn_ThreeAdv 2,3 40 'save_CAAE_LRW_RAE/GRU_unet_128_non_align_ThreeAdv'