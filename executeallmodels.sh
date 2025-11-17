#!/bin/bash

for dir in experimento*/; do
    if [ -d "$dir" ]; then
        echo "- Entrando na pasta: $dir"
        cd "$dir"
        python model_rf.py
        python model_lstm.py
        cd ..
    fi
done
