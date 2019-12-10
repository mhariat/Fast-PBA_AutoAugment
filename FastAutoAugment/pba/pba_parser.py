from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import collections
import json

from FastAutoAugment.pba.pba_augmentations import *

PbtUpdate = collections.namedtuple('PbtUpdate', [
    'target_trial_name', 'clone_trial_name', 'target_trial_epochs',
    'clone_trial_epochs', 'old_config', 'new_config'
])


def parse_policy(policy_emb):
    policy = []
    num_xform = NUM_HP_TRANSFORM
    xform_names = HP_TRANSFORM_NAMES
    assert len(policy_emb
               ) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(
                   len(policy_emb), 2 * num_xform)
    for i, xform in enumerate(xform_names):
        policy.append((xform, policy_emb[2 * i] / 10., policy_emb[2 * i + 1]))
    return policy


def parse_log(file_path, epochs):
    raw_policy_file = open(file_path, "r").readlines()
    raw_policy = []
    for line in raw_policy_file:
        try:
            raw_policy_line = json.loads(line)
        except:
            raw_policy_line = ast.literal_eval(line)
        raw_policy.append(raw_policy_line)

    # Depreciated use case has policy as list instead of dict config.
    for r in raw_policy:
        for i in [4, 5]:
            if isinstance(r[i], list):
                r[i] = {"hp_policy": r[i]}
    raw_policy = [PbtUpdate(*r) for r in raw_policy]
    policy = []

    # Sometimes files have extra lines in the beginning.
    to_truncate = None
    for i in range(len(raw_policy) - 1):
        if raw_policy[i][0] != raw_policy[i + 1][1]:
            to_truncate = i
    if to_truncate is not None:
        raw_policy = raw_policy[to_truncate + 1:]

    # Initial policy for trial_to_clone_epochs.
    policy.append([raw_policy[0][3], raw_policy[0][4]["hp_policy"]])

    current = raw_policy[0][3]
    for i in range(len(raw_policy) - 1):
        # End at next line's trial epoch, start from this clone epoch.
        this_iter = raw_policy[i + 1][3] - raw_policy[i][3]
        assert this_iter >= 0, (i, raw_policy[i + 1][3], raw_policy[i][3])
        assert raw_policy[i][0] == raw_policy[i + 1][1], (i, raw_policy[i][0],
                                                          raw_policy[i + 1][1])
        policy.append([this_iter, raw_policy[i][5]["hp_policy"]])
        current += this_iter

    # Last cloned trial policy is run for (end - clone iter of last logged line)
    policy.append([epochs - raw_policy[-1][3], raw_policy[-1][5]["hp_policy"]])
    current += epochs - raw_policy[-1][3]
    assert epochs == sum([p[0] for p in policy])
    return policy


def parse_log_schedule(file_path, epochs, multiplier=1):
    policy = parse_log(file_path, epochs)
    schedule = []
    count = 0
    for num_iters, pol in policy:
        for _ in range(int(num_iters * multiplier)):
            schedule.append(pol)
            count += 1
    for _ in range(int(epochs * multiplier) - count):
        schedule.append(policy[-1][1])
    return schedule


def get_policy(file_path, hp_epochs, epochs):
    multiplier = int(epochs/hp_epochs)
    raw_policy = parse_log_schedule(file_path, hp_epochs, multiplier)
    policy = []
    split = len(raw_policy[0]) // 2
    for pol in raw_policy:
        cur_pol = parse_policy(pol[:split])
        cur_pol.extend(parse_policy(pol[split:]))
        policy.append(cur_pol)
    return policy
