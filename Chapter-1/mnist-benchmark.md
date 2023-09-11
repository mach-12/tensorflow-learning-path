Below is a table benchmarking different runs, tracked with commits.


| EPOCHS | BATCH SIZE |     HIDDEN     | DROPOUT | TRAIN_ACC | VAL_ACC | TEST_ACC | TRAIN_LOSS | VAL_LOSS | TEST_LOSS | COMMIT |
|:------:|:----------:|:--------------:|:-------:|:---------:|:-------:|:--------:|------------|----------|-----------|:------:|
|   200  |     128    |       -        |    -    |  0.9231   | 0.9230  |  0.9225  | 0.2763     | 0.2754   | 0.2774    |        |
|   200  |     128    | {1:128, 2:128} |    -    |  0.9765   | 0.9754  |  0.9765  | 0.0166     | 0.0909   | 0.0825    |        |
|        |            |                |         |           |         |          |            |          |           |        |