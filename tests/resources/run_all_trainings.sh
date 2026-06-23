ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "${MTT_TESTS_PRERUN_TRAININGS:0}" = 1 ]; then
    echo "Running all trainings..."
    bash "$ROOT_DIR/run_trainings.sh" 32-bit
    bash "$ROOT_DIR/run_trainings.sh" 64-bit
    bash "$ROOT_DIR/run_trainings.sh" pet
else
    echo "Skipping prerunning trainings. Set MTT_TESTS_PRERUN_TRAININGS=1 to enable."
fi