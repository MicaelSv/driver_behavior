#!/bin/bash

for dir in experimento*/; do
    if [ -d "$dir" ]; then
        echo "- Entrando na pasta: $dir"
        cd "$dir"
        python mygraphics.py
        cd ..
    fi
done
