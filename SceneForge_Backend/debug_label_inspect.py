from transformers import AutoModelForSemanticSegmentation
m = AutoModelForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640', local_files_only=True)
labels = m.config.id2label
for k,v in labels.items():
    if any(x in v.lower() for x in ['chair','sofa','armchair','bench','seat','cushion']):
        print(k, v)
print('Total classes:', len(labels))