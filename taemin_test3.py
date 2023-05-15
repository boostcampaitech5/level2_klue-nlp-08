from transformers import AutoTokenizer
type_dict = {'ORG': '단체','PER':'사람','LOC':'지역','POH':'직업','NOH':'숫자','DAT':'날짜'}
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

print(tokenizer.vocab)