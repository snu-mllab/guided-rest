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

import random
import re
from dataclasses import dataclass
from typing import Any

from recipe.countdown.utils import apply_operation


@dataclass
class Node:
    name: str
    nums: list[int]
    operations: list[str]
    gen_start: int
    gen_end: int
    expl_start: int
    expl_end: int


def get_subgoal_nodes(search_path: str, extra_info: dict[str, Any]) -> list[Node]:
    # get extra info
    target = extra_info["target"]
    nums = extra_info["nums"]
    solution = extra_info["solution"]

    # match root node
    nodes = []
    operations = []
    target_str = re.escape(str(target))
    nums_str = re.escape(str(nums))
    operations_str = re.escape(str(operations))
    expl_pattern = rf"Current State: {target_str}:{nums_str}, Operations: {operations_str}\n"
    expl_matches = list(re.finditer(expl_pattern, search_path))
    if not expl_matches:
        return nodes
    expl_match = expl_matches[0]
    expl_start = expl_match.start()
    expl_end = expl_match.end()
    node = Node(
        name="0",
        nums=nums,
        operations=operations,
        gen_start=0,
        gen_end=0,
        expl_start=expl_start,
        expl_end=expl_end,
    )
    nodes.append(node)

    # match non-root nodes
    for operation in solution[:-1]:
        # update node info
        nums = apply_operation(nums, operation)
        operations = operations + [operation]

        # match generation line
        operation_str = re.escape(operation)
        nums_str = re.escape(str(nums))
        target_str = re.escape(str(target))
        gen_pattern = (
            rf"Exploring Operation: {operation_str}, Resulting Numbers: {nums_str}\n"
            rf"Generated Node #(\d+(?:,\d+)*): {target_str}:{nums_str} Operation: {operation_str}\n"
        )
        gen_matches = list(re.finditer(gen_pattern, search_path))
        if not gen_matches:
            return nodes
        gen_match = gen_matches[0]
        name = gen_match.group(1)
        gen_start = gen_match.start()
        gen_end = gen_match.end()

        # match exploration line
        name_str = re.escape(name)
        target_str = re.escape(str(target))
        nums_str = re.escape(str(nums))
        operations_str = re.escape(str(operations))
        expl_pattern = (
            rf"Moving to Node #{name_str}\n"
            rf"Current State: {target_str}:{nums_str}, Operations: {operations_str}\n"
        )
        expl_matches = list(re.finditer(expl_pattern, search_path[gen_end:]))
        if not expl_matches:
            return nodes
        expl_match = expl_matches[0]
        expl_start = gen_end + expl_match.start()
        expl_end = gen_end + expl_match.end()

        # get subgoal node
        node = Node(
            name=name,
            nums=nums,
            operations=operations,
            gen_start=gen_start,
            gen_end=gen_end,
            expl_start=expl_start,
            expl_end=expl_end,
        )
        nodes.append(node)

    return nodes


def get_child_nodes(search_path: str, parent_node: Node) -> list[Node]:
    # get node info
    parent_name = parent_node.name
    parent_expl_end = parent_node.expl_end

    # match generation line
    parent_name_str = re.escape(parent_name)
    gen_pattern = (
        rf"Exploring Operation: \d+[-+*/]\d+=\d+, Resulting Numbers: \[.*?\]\n"
        rf"Generated Node #({parent_name_str},\d+): \d+:\[.*?\] Operation: \d+[-+*/]\d+=\d+\n"
    )
    gen_matches = list(re.finditer(gen_pattern, search_path[parent_expl_end:]))

    # match child nodes
    nodes = []
    for gen_match in gen_matches:
        # match exploration line
        name = gen_match.group(1)
        gen_start = parent_expl_end + gen_match.start()
        gen_end = parent_expl_end + gen_match.end()
        name_str = re.escape(name)
        expl_pattern = (
            rf"Moving to Node #{name_str}\n"
            rf"Current State: \d+:\[.*?\], Operations: \[.*?\]\n"
        )
        expl_matches = list(re.finditer(expl_pattern, search_path[gen_end:]))
        if not expl_matches:
            continue
        expl_match = expl_matches[0]
        expl_start = gen_end + expl_match.start()
        expl_end = gen_end + expl_match.end()

        # get child node
        node = Node(
            name=name,
            nums=[],  # dummy
            operations=[],  # dummy
            gen_start=gen_start,
            gen_end=gen_end,
            expl_start=expl_start,
            expl_end=expl_end,
        )
        nodes.append(node)

    return nodes


def augment_search_path(search_path: str, extra_info: dict[str, Any]) -> str:
    # get extra info
    target = extra_info["target"]
    nums = extra_info["nums"]
    solution = extra_info["solution"]

    # get subgoal nodes
    subgoal_nodes = get_subgoal_nodes(search_path, extra_info)

    # reset search path if there are no subgoal nodes
    if not subgoal_nodes:
        search_path = f"Current State: {target}:{nums}, Operations: []\n"
        return search_path

    # get parent node
    parent_node = subgoal_nodes[-1]

    # get node info
    parent_name = parent_node.name
    parent_nums = parent_node.nums
    parent_operations = parent_node.operations
    parent_expl_end = parent_node.expl_end

    # update node info
    operation = solution[len(parent_operations)]
    nums = apply_operation(parent_nums, operation)
    operations = parent_operations + [operation]

    # add goal line if only one number is left
    if len(nums) == 1:
        search_path = (
            search_path[:parent_expl_end]
            + f"Exploring Operation: {operation}, Resulting Numbers: {nums}\n"
            + f"{nums[0]},{target} equal: Goal Reached"
        )
        return search_path

    # check if subgoal node is generated but not explored
    operation_str = re.escape(operation)
    nums_str = re.escape(str(nums))
    parent_name_str = re.escape(parent_name)
    target_str = re.escape(str(target))
    gen_pattern = (
        rf"Exploring Operation: {operation_str}, Resulting Numbers: {nums_str}\n"
        rf"Generated Node #({parent_name_str},\d+): {target_str}:{nums_str} Operation: {operation_str}\n"
    )
    gen_matches = list(re.finditer(gen_pattern, search_path[parent_expl_end:]))

    # add exploration line if subgoal node is generated
    if gen_matches:
        gen_match = gen_matches[0]
        name = gen_match.group(1)
        gen_end = parent_expl_end + gen_match.end()
        search_path = (
            search_path[:gen_end]
            + f"Moving to Node #{name}\n"
            + f"Current State: {target}:{nums}, Operations: {operations}\n"
        )
        return search_path

    # get child nodes
    child_nodes = get_child_nodes(search_path, parent_node)

    # add generation and exploration lines if no child nodes exist
    if not child_nodes:
        name = parent_name + ",0"
        search_path = (
            search_path[:parent_expl_end]
            + f"Exploring Operation: {operation}, Resulting Numbers: {nums}\n"
            + f"Generated Node #{name}: {target}:{nums} Operation: {operation}\n"
            + f"Moving to Node #{name}\n"
            + f"Current State: {target}:{nums}, Operations: {operations}\n"
        )
        return search_path

    # choose node
    node = random.choice(child_nodes)

    # replace generation and exploration lines
    name = node.name
    gen_start = node.gen_start
    gen_end = node.gen_end
    expl_start = node.expl_start
    search_path = (
        search_path[:gen_start]
        + f"Exploring Operation: {operation}, Resulting Numbers: {nums}\n"
        + f"Generated Node #{name}: {target}:{nums} Operation: {operation}\n"
        + search_path[gen_end:expl_start]
        + f"Moving to Node #{name}\n"
        + f"Current State: {target}:{nums}, Operations: {operations}\n"
    )
    return search_path
