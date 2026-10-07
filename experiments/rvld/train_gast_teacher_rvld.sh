#!/usr/bin/env bash
# RVLD student from a frozen GAST teacher: train_gast_teacher_rvld.sh <teacher_package> [iterations] [envs] [run]
exec bash "$(dirname "${BASH_SOURCE[0]}")/../common/distill_from_gast_teacher.sh" cnn_gru "$@"
