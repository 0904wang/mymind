from transformers import AutoTokenizer
# 指向你的目录
tokenizer = AutoTokenizer.from_pretrained("./model") 
print(f"EOS Token: {tokenizer.eos_token}") 
# 如果输出 <|endoftext|>，说明一切正常！