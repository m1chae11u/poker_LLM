# litgit_workspace.py

'''
custom dataset format:
{
    instruction: "yadadada",
    input: "",
    output: "The game is currently at the stage of [pre-flop]. My position is [CO], and my holding is [Nine of Diamond and Nine of Club]. My hand currently forms [one pair]. The current pot size is [130.5 chips], and my stack size left is [86.0 chips]. The stack-to-pot ratio is [low]. Given these information and the action history, my optimal decision is: fold."
}
'''

'''
download model you want to use:

litgpt download google/gemma-2b \
  --checkpoint_dir /data/michael_lu/poker_LLM/litgpt

CUDA_VISIBLE_DEVICES=3 litgpt finetune /data/michael_lu/poker_LLM/litgpt/google/gemma-2b \
  --data JSON \
  --data.json_path /home/michael_lu/poker_LLM/data/formatted_for_lit_gpt/lit_gpt_custom_setup \
  --out_dir /data/michael_lu/poker_LLM/litgpt
  
  or
  
litgpt finetune_full /data/michael_lu/poker_LLM/litgpt/google/gemma-2b \
  --data JSON \
  --data.json_path /home/michael_lu/poker_LLM/data/formatted_for_lit_gpt/lit_gpt_custom_setup \
  --out_dir /data/michael_lu/poker_LLM/litgpt

  or 

  litgpt finetune_lora /data/michael_lu/poker_LLM/litgpt/google/gemma-2b \
  --data JSON \
  --data.json_path /home/michael_lu/poker_LLM/data/formatted_for_lit_gpt/lit_gpt_custom_setup \
  --out_dir /data/michael_lu/poker_LLM/litgpt

You can also customize how the dataset is read by using these additional parameters

val_split_fraction: The fraction of the data to split. Defaults to 0.1

seed: The seed value to reproduce the same random splits for train and test data.

mask_inputs: Whether to mask the prompt section from the label (with ignore_index).

ignore_index: The index to use for labels that should be ignored. Defaults to -100 (used when mask_inputs is True).


Test your model
    
litgpt generate "/data/michael_lu/poker_LLM/litgpt/google/gemma-2b" \
  --prompt "\n\nYou are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.\n\nHere is a game summary:\n\nThe small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\nIn this hand, your position is CO, and your holding is [Queen of Heart and Queen of Club].\nBefore the flop, CO raise 2.3 chips, and BB call. Assume that all other players that is not mentioned folded.\nThe flop comes Nine Of Spade, Seven Of Spade, and Four Of Club, then BB check, and CO check.\nThe turn comes Ace Of Heart, then BB bet 6 chips, and CO call.\nThe river comes Eight Of Club, then BB check.\nYou currently have One Pair(One Pair, Queens with Ace, Nine, Eight kickers).\n\nNow it is your turn to make a move.\nTo remind you, the current pot size is 17.0 chips, and your holding is [Queen of Heart and Queen of Club]. You currently have One Pair.\n\nDecide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.\nYour optimal action is:"

  
litgpt chat /data/michael_lu/poker_LLM/litgpt/step-044000
'''


from litgpt import LLM
llm = LLM.load("/data/michael_lu/poker_LLM/litgpt/checkpoints/google/gemma-2b")
text = llm.generate("\n\nYou are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.\n\nHere is a game summary:\n\nThe small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\nIn this hand, your position is CO, and your holding is [Queen of Heart and Queen of Club].\nBefore the flop, CO raise 2.3 chips, and BB call. Assume that all other players that is not mentioned folded.\nThe flop comes Nine Of Spade, Seven Of Spade, and Four Of Club, then BB check, and CO check.\nThe turn comes Ace Of Heart, then BB bet 6 chips, and CO call.\nThe river comes Eight Of Club, then BB check.\nYou currently have One Pair(One Pair, Queens with Ace, Nine, Eight kickers).\n\nNow it is your turn to make a move.\nTo remind you, the current pot size is 17.0 chips, and your holding is [Queen of Heart and Queen of Club]. You currently have One Pair.\n\nDecide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.\nYour optimal action is:")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.       


'''
setup:

litgpt download google/gemma-2b \
  --access_token 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 \
  --checkpoint_dir /data/michael_lu/poker_LLM/litgpt/checkpoints

for screen: lora_SFT

CUDA_VISIBLE_DEVICES=3 litgpt finetune /data/michael_lu/poker_LLM/litgpt/checkpoints/google/gemma-2b \
  --data JSON \
  --data.json_path /home/michael_lu/poker_LLM/data/formatted_for_lit_gpt/lit_gpt_custom_setup/combined_train_set.json \
  --data.val_split_fraction 0.0001 \
  --out_dir /data/michael_lu/poker_LLM/litgpt/lora_out \
  --train.save_interval 500 \
  --train.log_interval 50 \
  --train.epochs 3 \
  --train.global_batch_size 16 \
  --train.micro_batch_size 8 \
  --eval.interval 200 \
  --eval.initial_validation False

–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

for screen: testing

CUDA_VISIBLE_DEVICES=7 litgpt finetune /data/michael_lu/poker_LLM/litgpt/checkpoints/google/gemma-2b \
  --data JSON \
  --data.json_path /home/michael_lu/poker_LLM/data/formatted_for_lit_gpt/lit_gpt_custom_setup/combined_train_set.json \
  --data.val_split_fraction 0.0001 \
  --out_dir /data/michael_lu/poker_LLM/litgpt/test_out \
  --train.save_interval 500 \
  --train.log_interval 50 \
  --train.epochs 3 \
  --train.max_steps 5 \
  --train.global_batch_size 16 \
  --train.micro_batch_size 8 \
  --eval.interval 200 \
  --eval.initial_validation False

for screen: FULL_finetuning

CUDA_VISIBLE_DEVICES=4 litgpt finetune_full /data/michael_lu/poker_LLM/litgpt/checkpoints/google/gemma-2b \
  --data JSON \
  --data.json_path /home/michael_lu/poker_LLM/data/formatted_for_lit_gpt/lit_gpt_custom_setup/combined_train_set.json \
  --data.val_split_fraction 0.0001 \
  --out_dir /data/michael_lu/poker_LLM/litgpt/full_finetune_out \
  --train.save_interval 500 \
  --train.log_interval 50 \
  --train.epochs 3 \
  --train.max_steps 5 \
  --train.global_batch_size 16 \
  --eval.interval 200 \
  --eval.initial_validation False

temp

CUDA_VISIBLE_DEVICES=4 litgpt finetune full /data/michael_lu/poker_LLM/litgpt/checkpoints/google/gemma-2b \
  --data JSON \
  --data.json_path /home/michael_lu/poker_LLM/data/formatted_for_lit_gpt/lit_gpt_custom_setup/combined_train_set.json \
  --data.val_split_fraction 0.0001 \
  --out_dir /data/michael_lu/poker_LLM/litgpt/full_finetune_out \
  --train.save_interval 500 \
  --train.log_interval 50 \
  --train.epochs 3 \
  --train.global_batch_size 16 \
  --eval.interval 200 \
  --eval.initial_validation False \
  --devices 1
'''
# --train.max_steps 5 (temporary parameter used to validate)