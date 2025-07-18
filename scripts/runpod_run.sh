GIT_ROOT=$(git rev-parse --show-toplevel)
source $HOME/.local/bin/env

# List config files in $GIT_ROOT/config and prompt user to select one
config_files=($(ls $GIT_ROOT/config))
echo "Available config files:"
for i in "${!config_files[@]}"; do
    echo "$((i+1)): ${config_files[i]}"
done
read -p "Enter the number of the config file to use: " config_num
config_file=${config_files[$((config_num-1))]}

uv run python $GIT_ROOT/train.py -f $GIT_ROOT/config/$config_file
