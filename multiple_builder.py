# -*- coding:utf-8 -*-
try:
    from typing import List, Tuple, Dict, Any, Optional
except ImportError:
    pass

import Rhino.Geometry as geo  # type: ignore
import scriptcontext as sc  # type: ignore
import Rhino  # type: ignore
import utils
import facade_plan
import facade_generator as fg
import importlib
import random
import time

# 모듈 새로고침
importlib.reload(fg)
importlib.reload(utils)
importlib.reload(facade_plan)

# facade_plan에서 Facade 클래스를 가져옴
from facade_plan import Facade


class BuildingGroup:
    """개별 빌딩 그룹을 나타내는 클래스"""

    def __init__(self, breps: List[geo.Brep], group_id: int):
        self.breps = breps
        self.group_id = group_id


class MultipleBuilder:
    """여러 빌딩에 대해 각각 다른 파사드 타입을 적용하는 클래스"""

    def __init__(
        self,
        building_breps: List[geo.Brep],
        base: fg.FGInputs,
        facade_types: Optional[List[int]] = None,
        grouping_tolerance: float = 0.1,
        variation_factor: float = 0.0,
    ):
        # 기본/공통 입력
        base = base.coerce()
        self.building_breps = building_breps
        self.floor_height = float(base.floor_height)
        self.grouping_tolerance = float(grouping_tolerance)
        self.variation_factor = float(variation_factor)
        self.bake_block = bool(base.bake_block)

        # 빌딩을 그룹화
        self.building_groups = self._group_buildings()
        self.num_groups = len(self.building_groups)

        # 랜덤 시드 설정 (매번 다른 결과를 위해)
        random.seed(int(time.time() * 1000) % 10000)

        # 파사드 타입은 리스트에서 랜덤 선택 풀로 사용 (미지정 시 base.facade_type 사용)
        self.facade_type_options = (
            facade_types if facade_types else [int(base.facade_type)]
        )

        # 파라미터들을 그룹 수에 맞게 조정하고 랜덤 변동 적용
        # 이제 외부에서 리스트를 받지 않고 base 값 하나만 받아 내부에서 확장
        self.group_pattern_lengths = self._prepare_group_parameters(
            None, float(base.pattern_length)
        )
        self.group_pattern_depths = self._prepare_group_parameters(
            None, float(base.pattern_depth)
        )
        self.group_pattern_ratios = self._prepare_group_parameters(
            None, float(base.pattern_ratio)
        )
        self.group_facade_offsets = self._prepare_group_parameters(
            None, float(base.facade_offset)
        )
        self.group_slab_heights = self._prepare_group_parameters(
            None, float(base.slab_height)
        )
        self.group_slab_offsets = self._prepare_group_parameters(
            None, float(base.slab_offset)
        )

    def _prepare_group_parameters(
        self, param_list: Optional[List], default_value
    ) -> List:
        """파라미터 리스트를 그룹 수에 맞게 정규화하고 랜덤 변동을 적용하는 메서드"""
        # 1단계: 파라미터를 그룹 수에 맞게 정규화
        if param_list is None:
            base_params = [default_value] * self.num_groups
        elif len(param_list) < self.num_groups:
            # 부족한 경우 마지막 값으로 채움
            last_value = param_list[-1] if param_list else default_value
            base_params = param_list + [last_value] * (
                self.num_groups - len(param_list)
            )
        else:
            # 초과하는 경우 자름
            base_params = param_list[: self.num_groups]

        # 2단계: 각 그룹별로 랜덤 변동 적용
        if self.variation_factor <= 0:
            return base_params

        varied_params = []
        for base_value in base_params:
            # base_value * (1.0 ~ 1.0 + variation_factor) 범위에서 랜덤
            min_multiplier = 1.0
            max_multiplier = 1.0 + self.variation_factor
            multiplier = random.uniform(min_multiplier, max_multiplier)
            varied_params.append(base_value * multiplier)

        return varied_params

    def _get_random_facade_types(self) -> List[int]:
        """각 그룹별로 파사드 타입을 랜덤하게 선택하는 메서드"""
        return [random.choice(self.facade_type_options) for _ in range(self.num_groups)]

    def _group_buildings(self) -> List[BuildingGroup]:
        """빌딩들을 거리 기반으로 그룹화하는 메서드"""
        if not self.building_breps:
            return []

        groups = []
        assigned = [False] * len(self.building_breps)
        group_id = 0

        for i, brep in enumerate(self.building_breps):
            if assigned[i]:
                continue

            # 새 그룹 시작
            current_group = [brep]
            assigned[i] = True

            # 같은 그룹에 속할 브렙들 찾기
            for j, other_brep in enumerate(self.building_breps):
                if i == j or assigned[j]:
                    continue

                if self._are_buildings_close(brep, other_brep):
                    current_group.append(other_brep)
                    assigned[j] = True

            groups.append(BuildingGroup(current_group, group_id))
            group_id += 1

        return groups

    def _are_buildings_close(self, brep1: geo.Brep, brep2: geo.Brep) -> bool:
        """두 빌딩이 가까운지 판단하는 메서드 (거리 기반)"""
        try:
            # 각 brep의 중심점 계산
            bbox1 = brep1.GetBoundingBox(True)
            bbox2 = brep2.GetBoundingBox(True)

            center1 = geo.Point3d(
                (bbox1.Min.X + bbox1.Max.X) / 2,
                (bbox1.Min.Y + bbox1.Max.Y) / 2,
                (bbox1.Min.Z + bbox1.Max.Z) / 2,
            )

            center2 = geo.Point3d(
                (bbox2.Min.X + bbox2.Max.X) / 2,
                (bbox2.Min.Y + bbox2.Max.Y) / 2,
                (bbox2.Min.Z + bbox2.Max.Z) / 2,
            )

            # XY 평면에서의 거리만 고려 (층별로 분해된 경우 Z축은 무시)
            distance = (
                (center1.X - center2.X) ** 2 + (center1.Y - center2.Y) ** 2
            ) ** 0.5

            return distance <= self.grouping_tolerance

        except Exception as e:
            print(f"Error calculating distance between buildings: {e}")
            return False

    def generate_all_facades(
        self, pattern_types: List[int] = None
    ) -> List[List[Facade]]:
        """모든 빌딩 그룹에 대해 층별 Facade 리스트를 생성하여 그룹별로 반환"""
        if pattern_types is None:
            pattern_types = [1] * self.num_groups
        else:
            # pattern_types도 동일한 정규화 적용 (단, 랜덤 변동은 제외)
            if len(pattern_types) < self.num_groups:
                last_value = pattern_types[-1] if pattern_types else 1
                pattern_types = pattern_types + [last_value] * (
                    self.num_groups - len(pattern_types)
                )
            else:
                pattern_types = pattern_types[: self.num_groups]

        # 각 그룹별로 파사드 타입 미리 결정
        group_facade_types = self._get_random_facade_types()

        all_group_facades: List[List[Facade]] = []

        for i, building_group in enumerate(self.building_groups):
            try:
                # 그룹 내 각 Brep에 대해 개별적으로 파사드 생성 (같은 그룹은 동일한 파라미터 사용)
                group_facades: List[Facade] = []
                for brep in building_group.breps:
                    inputs = fg.FGInputs(
                        building_brep=brep,
                        floor_height=self.floor_height,
                        facade_type=group_facade_types[i],
                        pattern_length=self.group_pattern_lengths[i],
                        pattern_depth=self.group_pattern_depths[i],
                        pattern_ratio=self.group_pattern_ratios[i],
                        facade_offset=self.group_facade_offsets[i],
                        slab_height=self.group_slab_heights[i],
                        slab_offset=self.group_slab_offsets[i],
                        bake_block=self.bake_block,
                    )
                    facade_generator = fg.FacadeGenerator(inputs)

                    # 개별 Brep의 파사드 생성
                    floor_facades = facade_generator.generate(pattern_types[i])
                    # floor_facades: List[Facade] (또는 빈 리스트 - 블록 모드)
                    if floor_facades:
                        group_facades.extend(floor_facades)

                all_group_facades.append(group_facades)

            except Exception as e:
                print(f"[MultipleBuilder] Error in group loop {i+1}: {e}")
                all_group_facades.append([])

        return all_group_facades

    def get_combined_facade(self, pattern_types: List[int] = None) -> List[Facade]:
        """모든 그룹의 층별 Facade를 하나의 리스트로 평탄화하여 반환"""
        group_facade_lists = self.generate_all_facades(pattern_types)
        combined: List[Facade] = []
        for facades in group_facade_lists:
            combined.extend(facades or [])
        return combined

    def get_group_info(self) -> List[Dict[str, Any]]:
        """그룹화 정보를 반환하는 메서드"""
        info = []
        for group in self.building_groups:
            info.append(
                {
                    "group_id": group.group_id,
                    "brep_count": len(group.breps),
                    "breps": group.breps,  # 개별 Brep들 그대로 반환
                }
            )
        return info

    @classmethod
    def from_inputs(
        cls,
        building_breps: List[geo.Brep],
        base: fg.FGInputs,
        *,
        facade_types: Optional[List[int]] = None,
        grouping_tolerance: float = 0.1,
        variation_factor: float = 0.0,
    ) -> "MultipleBuilder":
        """FGInputs 하나로 MultipleBuilder 생성 (내부에서 그룹 파라미터 확장)"""
        return cls(
            building_breps,
            base,
            facade_types=facade_types,
            grouping_tolerance=grouping_tolerance,
            variation_factor=variation_factor,
        )

    @classmethod
    def from_globals(cls, globs: dict):
        base = fg.FGInputs.from_globals(globs)
        building_breps = globs.get("building_breps", [])
        grouping_tolerance = float(globs.get("grouping_tolerance", 0.1))
        variation_factor = float(globs.get("variation_factor", 0.0))
        facade_types = globs.get("facade_types", None)
        mb = cls.from_inputs(
            building_breps,
            base,
            facade_types=facade_types,
            grouping_tolerance=grouping_tolerance,
            variation_factor=variation_factor,
        )
        pattern_types = globs.get("pattern_types", None)
        return mb, pattern_types


def _flatten(fs: List[Facade]):
    g, w, f, s = [], [], [], []
    for fc in fs or []:
        g.extend(fc.glasses or [])
        w.extend(fc.walls or [])
        f.extend(fc.frames or [])
        s.extend(fc.slabs or [])
    return g, w, f, s


# Grasshopper 컴포넌트 입력이 있을 때만 실행되도록 안전 가드
if __name__ == "__main__" or "building_breps" in globals():
    mb, _pattern_types = MultipleBuilder.from_globals(globals())

    try:
        group_facades = mb.generate_all_facades(_pattern_types)
        facades = mb.get_combined_facade(_pattern_types)
        group_info = mb.get_group_info()

        glasses, walls, frames, slabs = _flatten(facades)
        num_groups = len(group_info)
        group_brep_counts = [info["brep_count"] for info in group_info]
    except Exception as e:
        print(f"Error in MultipleBuilder: {e}")
        glasses = []
        walls = []
        frames = []
        slabs = []
        num_groups = 0
        group_brep_counts = []
