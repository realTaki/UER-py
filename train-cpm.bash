# python pretrain.py --dataset_path lyric_dataset.pt \
#                     --pretrained_model_path models/clyric/lyric_model.bin \
#                     --vocab_path models/clyric/vocab.txt \
#                     --config_path models/clyric/config.json \
#                     --output_model_path models/clyric/lyric_gpt2_model.bin \
#                     --world_size 1 --gpu_ranks 0 \
#                     --total_steps 100000 --save_checkpoint_steps 10000 --report_steps 100 \
#                     --learning_rate 5e-5 --batch_size 8 \
#                     --embedding word_pos --remove_embedding_layernorm \
#                     --encoder transformer --mask causal --layernorm_positioning pre \
#                     --target lm --tie_weights

python3 pretrain.py --dataset_path datasets/poems/cpm_256_dataset.pt \
                    --pretrained_model_path models/cpm/dis-cpm-uer.bin \
                    --spm_model_path models/cpm/spiece.model \
                    --tokenizer cpm \
                    --config_path models/cpm-config.json \
                    --output_model_path models/cpm/cpm.bin \
                    --world_size 1 --gpu_ranks 0 \
                    --total_steps 120000 --save_checkpoint_steps 10000 --report_steps 1000 \
                    --learning_rate 5e-5 --batch_size 24
