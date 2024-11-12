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
litgpt finetune path_to_gemma_2b_weights \
  --data JSON \
  --data.json_path poker_dataset.json \
  --data.val_split_fraction 0.1 \
  --out_dir out/custom-poker-model
'''


from litgpt import LLM
llm = LLM.load("microsoft/phi-2")
text = llm.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.       