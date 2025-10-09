# Copyright 2025 SNU MLLAB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def apply_operation(nums: list[int], operation: str) -> list[int]:
    op_pattern = r"(\d+)([-+*/])(\d+)=(\d+)"
    op_match = re.match(op_pattern, operation)
    if not op_match:
        raise ValueError(f"Invalid operation: {operation}")
    a = int(op_match.group(1))
    op = op_match.group(2)
    b = int(op_match.group(3))
    c = int(op_match.group(4))
    if eval(f"{a} {op} {b} != {c}"):
        raise ValueError(f"Invalid operation: {operation}")
    new_nums = nums.copy()
    new_nums.remove(a)
    new_nums.remove(b)
    new_nums.append(c)
    return new_nums


def grade_search_path(search_path: str, target: int, nums: list[int]) -> bool:
    # match goal line
    goal_pattern = r"\d+,\d+ equal: Goal Reached"
    goal_matches = list(re.finditer(goal_pattern, search_path))
    if not goal_matches:
        return False
    goal_match = goal_matches[0]
    goal_end = goal_match.end()

    # match generation line
    gen_pattern = r"Exploring Operation: (\d+[-+*/]\d+=\d+), Resulting Numbers: \[.*?\]\n"
    gen_matches = list(re.finditer(gen_pattern, search_path[:goal_end]))
    if not gen_matches:
        return False
    gen_match = gen_matches[-1]
    operation = gen_match.group(1)

    # match exploration line
    expl_pattern = r"Current State: \d+:\[.*?\], Operations: (\[.*?\])\n"
    expl_matches = list(re.finditer(expl_pattern, search_path[:goal_end]))
    if not expl_matches:
        return False
    expl_match = expl_matches[-1]
    operations = eval(expl_match.group(1))

    # apply operations
    operations += [operation]
    for operation in operations:
        nums = apply_operation(nums, operation)
    if len(nums) != 1:
        return False

    # verify result
    is_correct = nums[0] == target
    return is_correct
