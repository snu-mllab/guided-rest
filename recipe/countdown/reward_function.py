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

from typing import Any

from recipe.countdown.utils import grade_search_path


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any]) -> float:
    target = extra_info["target"]
    nums = extra_info["nums"]
    try:
        is_correct = grade_search_path(solution_str, target, nums)
        return 1.0 if is_correct else 0.0
    except:
        return 0.0
