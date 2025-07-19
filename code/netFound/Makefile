PREPROCESS_SRC_DIR = src/pre_process/packets_processing_src
PREPROCESS_BUILD_DIR = build/preprocess
DATA_DIR = data/test

# Targets
all: clean compile preprocess pretrain finetune

clean:
	rm -rf $(PREPROCESS_BUILD_DIR)
	rm -f src/pre_process/1_filter
	rm -f src/pre_process/3_field_extraction
	rm -rf $(DATA_DIR)/pretraining/split $(DATA_DIR)/pretraining/filtered $(DATA_DIR)/pretraining/extracted $(DATA_DIR)/pretraining/final
	rm -rf $(DATA_DIR)/finetuning/split $(DATA_DIR)/finetuning/filtered $(DATA_DIR)/finetuning/extracted $(DATA_DIR)/finetuning/final

compile:
	mkdir -p $(PREPROCESS_BUILD_DIR)
	cmake -S $(PREPROCESS_SRC_DIR) -B $(PREPROCESS_BUILD_DIR)
	make -C $(PREPROCESS_BUILD_DIR)
	cp $(PREPROCESS_BUILD_DIR)/1_filter src/pre_process/1_filter
	cp $(PREPROCESS_BUILD_DIR)/3_field_extraction src/pre_process/3_field_extraction

preprocess:
	python3 ./scripts/preprocess_data.py --input_folder $(DATA_DIR)/pretraining --action pretrain --tokenizer_config configs/TestPretrainingConfig.json --combined
	python3 ./scripts/preprocess_data.py --input_folder $(DATA_DIR)/finetuning --action finetune --tokenizer_config configs/TestFinetuningConfig.json --combined

pretrain:
	./scripts/pretrain-test.sh

finetune:
	./scripts/finetune-test.sh


.PHONY: all clean compile preprocess pretrain finetune