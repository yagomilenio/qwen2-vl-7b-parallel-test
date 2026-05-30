#!/bin/bash

LMS_BIN="$HOME/.lmstudio/bin/lms"

if $LMS_BIN server status 2>/dev/null | grep -q "running"; then
    echo "LMStudio server ya está corriendo"
else
    echo "Arrancando LMStudio server..."
    $LMS_BIN server start
    for i in $(seq 1 10); do
        if $LMS_BIN server status 2>/dev/null | grep -q "running"; then
            echo "Servidor listo"
            break
        fi
        echo "Esperando... ($i/10)"
        sleep 2
    done
fi
