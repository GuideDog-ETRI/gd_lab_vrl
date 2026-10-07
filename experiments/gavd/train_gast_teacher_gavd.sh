#!/usr/bin/env bash
# GAVD student from a frozen GAST teacher: train_gast_teacher_gavd.sh <teacher_package> [iterations] [envs] [run]
exec bash "$(dirname "${BASH_SOURCE[0]}")/../common/distill_from_gast_teacher.sh" grid_attention_v1 "$@"
