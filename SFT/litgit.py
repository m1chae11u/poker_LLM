# litgit.py

# task: we want to do supervised instruction finetuning

'''
custom poker dataset:
{
    instruction: "yadadada",
    input: "",
    output: ""The game is currently at the stage of [pre-flop]. My position is [CO], and my holding is [Nine of Diamond and Nine of Club]. My hand currently forms [one pair]. The current pot size is [130.5 chips], and my stack size left is [86.0 chips]. The stack-to-pot ratio is [low]. Given these information and the action history, my optimal decision is: fold.""
}
'''
# step 1:

'''
litgpt finetune_full path_to_gemma_2b_weights \
  --data JSON \
  --data.json_path path/to/your/data.json \
  --data.val_split_fraction 0.1 \
  --out_dir out/custom-poker-model


  or 


  litgpt finetune_lora tiiuae/falcon-7b \
  --data JSON \
  --data.json_path path/to/your/data.json \
  --data.val_split_fraction 0.1

You can also customize how the dataset is read by using these additional parameters

val_split_fraction: The fraction of the data to split. Defaults to 0.1

seed: The seed value to reproduce the same random splits for train and test data.

mask_inputs: Whether to mask the prompt section from the label (with ignore_index).

ignore_index: The index to use for labels that should be ignored. Defaults to -100 (used when mask_inputs is True).
'''


from litgpt import LLM
llm = LLM.load("microsoft/phi-2")
text = llm.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.       