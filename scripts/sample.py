import torch
from setting import BASE_MODEL_NAME, PEFT_NAME, TOKENIER_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIER_NAME, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    if torch.cuda.is_available():
        model = model.to("cuda")

    while True:
        print(">", end="")
        text = input()
        token_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=100,
                min_new_tokens=100,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        output = tokenizer.decode(output_ids.tolist()[0])
        print(output)


if __name__ == "__main__":
    main()
"""
西田幾多郎は、その著書「自覚の哲学」の中で、次のように書きました。  
「知識を、自分のものと考えることに満足していると、自己の限界に目覚めることを忘れてしまう。しかし、他者との協同なしには、自己の本当の理解に達することはできないのだ。知識は他者と相互の、協同の力によってこそ、得られるのである。」(引用終わり)  
この一節を、私たちは今から学び直すべきです。そして、これからの社会をリードする子どもたちに、その能力を伸ばすべく、
"""
