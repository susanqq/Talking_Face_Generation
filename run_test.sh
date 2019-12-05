test_cnn(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore test.py  \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --ckpt $2'/model_G18.pt' \
  --save_dir $2 \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}

test_rnn(){
  CUDA_VISIBLE_DEVICES=$1 python  -W ignore test.py  \
  --num_input_imgs 1 \
  --size_image 128 \
  --audio_encoder 'reduce' \
  --img_encoder 'reduce' \
  --img_decoder 'reduce' \
  --rnn_type 'GRU' \
  --ckpt $2'/model_final.pt' \
  --save_dir $2 \
  --test_dir '/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_TCD' \
  --num_output_length 512 \
  --is_train True \
  --if_tanh False \
  --gpu $1
}


#test_cnn 2 'save_CAAE_VOX_CAE/face_unet_reduce_128_ThreeAdv'


#test_rnn 2 'save_CAAE_VOX_RAE_ia/face_unet_reduce_128_ThreeAdv'

test_rnn 2 'save_CAAE_TCD_RAE_ia/GRU_unet_seq'
#'testing/test_LRW'
#'/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_Vox_gen2'
#'/SLP_Extended/susan.s/Documents/Speech2Vid/data/test_TCD'