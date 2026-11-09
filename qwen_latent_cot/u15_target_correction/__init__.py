"""Isochronic state-flow editing on the frozen SenseNova-U1.5 stack."""

KIND = "u15_isochronic_state_flow_v1"
# v2: paired images share one scene field of view, and inverted caches are
# gated on both source and target reconstruction. Caches tagged v1 are invalid.
PROTOCOL_VERSION = "u15_isochronic_state_flow_v3"

LINEAR_PAIR = "linear_pair"
INVERTED_PAIR = "inverted_pair"
NATIVE_PARENT = "native_parent"
SOURCE_KINDS = (LINEAR_PAIR, INVERTED_PAIR, NATIVE_PARENT)

# Training roles. They are sampled explicitly so the task does not change with
# the cache acceptance rate.
ROLE_L_EDIT = "L_edit"
ROLE_I_EDIT = "I_edit"
ROLE_R_EDIT = "R_edit"
ROLE_R_KEEP = "R_keep"
# Sub-roles exist because the losses differ per task, not only per source domain:
# a repaired real failure and a directly specified edit both land in R_edit, and a
# forward pair and its reversed counterpart both land in L_edit, yet they must be
# weighted and supervised separately.
ROLE_L_REVERSE = "L_reverse"
ROLE_R_REPAIR = "R_repair"
# Active edit: an instructed change on a real history state, picked from what the
# completion already shows. It does not need the base model to have failed.
ROLE_R_ACTIVE = "R_active"
ROLES = (ROLE_L_EDIT, ROLE_I_EDIT, ROLE_R_EDIT, ROLE_R_KEEP, ROLE_L_REVERSE, ROLE_R_REPAIR, ROLE_R_ACTIVE)
# Which roles are trained through the frozen tail (terminal image supervision).
TERMINAL_ROLES = frozenset({ROLE_R_EDIT, ROLE_R_REPAIR, ROLE_R_KEEP, ROLE_R_ACTIVE})

CONDITION_PROTOCOL = "isochronic_condition_v1"
CACHE_KIND = "u15_isochronic_state_pairs_v1"

# Paired files must cover the same scene extent. Different aspect ratios are
# only accepted when the pair declares its crops explicitly.
GEOMETRY_PROTOCOL = "shared_field_of_view_v1"

# State labels are produced by inverting the *deployment* Euler map, not by
# integrating another solver backwards. Caches made with v1/v2 solvers mix a
# numerical mismatch into the label and cannot be reused.
INVERSION_PROTOCOL = "discrete_euler_fixpoint_v1"
