ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$ROOT_DIR/run_trainings.sh" 32-bit
bash "$ROOT_DIR/run_trainings.sh" 64-bit
bash "$ROOT_DIR/run_trainings.sh" pet