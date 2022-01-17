python prepare_big_data_for_punctuation_capitalization_task_simple.py \
  --output_dir /media/apeganov/DATA/punctuation_and_capitalization/all_punc_no_u/europarl_x3_92_128_12.12.2021 \
  --corpus_types europarl \
  --create_model_input \
  --bert_labels \
  --autoregressive_labels \
  --sequence_length_range 3 128 \
  --allowed_punctuation '.,?"-;:!()' \
  --no_label_if_all_characters_are_upper_case \
  --input_files ~/data/europarl/v10/training-monolingual/europarl-v10.en.tsv \
  --num_jobs 24 \
  --num_passes_through_dataset 3 \
  --dev_size 10000 \
  --test_size 0