"""A local instruction-tuned model answering many chats at once: the simulated seeker and the judge model.

Usage:
    model = ChatModel("Qwen/Qwen2.5-7B-Instruct")
    replies = model([[{"role": "user", "content": "Hello"}]], seed=0)
"""

import torch


class ChatModel:
    def __init__(self, name: str, device=None, batch_size=8, temperature=0.7, top_p=0.9, max_new_tokens=80):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.batch_size = batch_size
        self.temperature, self.top_p = temperature, top_p  # used when sampling
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def __call__(self, chats: list[list[dict]], seed: int, sample: bool = True, max_new_tokens=None) -> list[str]:
        """The next assistant message for each chat (a list of {"role", "content"} messages). Sampling is
        seeded per batch; sample=False decodes greedily."""
        out = []
        for start in range(0, len(chats), self.batch_size):
            texts = [
                self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in chats[start : start + self.batch_size]
            ]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            torch.manual_seed(seed + start)
            settings = (
                {"do_sample": True, "temperature": self.temperature, "top_p": self.top_p}
                if sample
                else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
            )
            gen = self.model.generate(
                **enc,
                **settings,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            out += [self.tokenizer.decode(g[enc.input_ids.shape[1] :], skip_special_tokens=True).strip() for g in gen]
        return out
