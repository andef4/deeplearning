#!/bin/bash
if [ ! -z "$PASSWORD" ]; then
    mkdir -p /home/user/.jupyter/
    echo "c.NotebookApp.password = '$PASSWORD'" > /home/user/.jupyter/jupyter_notebook_config.py
fi

if [ ! -z "$USER_ID" ]; then
    echo "Setting user id to $USER_ID"
    usermod -u $USER_ID user
fi

if [ ! -z "$GROUP_ID" ]; then
    echo "Setting group id to $GROUP_ID"
    groupmod -g $GROUP_ID user
fi

mkdir -p /home/user/code
cd /home/user/code

exec su user -c "$@"
