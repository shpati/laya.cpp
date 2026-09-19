# laya.cpp

C/C++ port of NandhaKishorM/laya

Runs on CPU. Windows, Linux, Mac OS. 



### Dependencies:

Copy the following files to your directory of choice (eg. laya.cpp): 

>  laya.cpp
> |-- config.json                    # copy from [convaiinnovations/laya/encoder](https://huggingface.co/convaiinnovations/laya/resolve/main/encoder/config.json)
> |-- example.json
> |-- laya.c
> |-- README.md
> |-- model.safetensors      # copy from [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya/resolve/main/model.safetensors)
> |-- rl_agent_config.json    # copy from [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya/resolve/main/rl_agent_config.json)
> |-- tokenizer.json              # copy from [convaiinnovations/laya/tokenizer](https://huggingface.co/convaiinnovations/laya/resolve/main/tokenizer/tokenizer.json)



### Build:

```
cd laya.cpp
cc -O3 -march=native -fopenmp laya.c -o laya -lm    (on Linux or Windows)
cc -O3 -march=native laya.c -o laya -lm             (on Mac OS)
```



### Run:

```
./laya MODEL_DIR input.json          ("-" reads the input from stdin)
./laya MODEL_DIR --tokenize "text"   (print token ids)
./laya MODEL_DIR --list-tensors
```



### Example:

```
./laya . example.json                (runs example on Linux, Mac OS)
laya.exe . example.json              (runs example on Windows)
```
