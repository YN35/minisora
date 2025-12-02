from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

model_name = "Qwen/Qwen3-4B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen3ForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=16384)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
