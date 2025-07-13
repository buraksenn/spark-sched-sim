GIT_ROOT=$(git rev-parse --show-toplevel)
source $HOME/.local/bin/env

uv run python $GIT_ROOT/train.py -f $GIT_ROOT/config/decima_tpch.yaml
