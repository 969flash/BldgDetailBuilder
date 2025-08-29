# -*- coding:utf-8 -*-
try:
    from typing import List, Tuple, Dict, Any, Optional
except ImportError:
    pass

import Rhino.Geometry as geo  # type: ignore
import scriptcontext as sc  # type: ignore
import Rhino  # type: ignore
import ghpythonlib.components as ghcomp  # type: ignore
import utils
import facade_plan
import importlib

# 모듈 새로고침
importlib.reload(utils)
importlib.reload(facade_plan)

# facade_plan에서 Facade 클래스를 가져옴
from facade_plan import Facade

####
# input : 3D Building  Brep
# output : Building Facade Breps
####


class FacadeGenerator:
    def __init__(
        self,
        building_brep: geo.Brep,
        floor_height: float,
        facade_type: int = 1,
        pattern_length: float = 4.0,
        pattern_depth: float = 1.0,
        pattern_ratio: float = 0.8,
        facade_offset: float = 0.2,
        slab_height: float = 0.0,
        slab_offset: float = 0.0,
    ):
        self.building_brep = building_brep
        self.building_curve = utils.get_outline_from_closed_brep(
            building_brep, geo.Plane.WorldXY
        )
        self.building_curve = utils.offset_regions_outward(
            self.building_curve, facade_offset
        )[0]

        self.building_height = self._get_building_height()
        # 층고 및 층수 계산 (반올림 사용). 유효하지 않은 층고 입력은 1층 처리
        self.floor_height = float(floor_height)
        if self.floor_height > 0:
            self.building_floor = max(
                1, int(round(self.building_height / self.floor_height))
            )
        else:
            self.building_floor = 1

        # 파사드 타입 및 패턴 파라미터
        self.facade_type = int(facade_type)
        self.pattern_length = float(pattern_length)
        self.pattern_depth = float(pattern_depth)
        self.pattern_ratio = float(pattern_ratio)

        # 슬래브 파라미터
        self.slab_height = float(slab_height)
        self.slab_offset = float(slab_offset)

    def _get_building_height(self) -> float:
        # 건물의 높이를 계산하는 로직을 구현합니다.
        bbox = self.building_brep.GetBoundingBox(True)
        return bbox.Max.Z - bbox.Min.Z

    def generate(self, pattern_type: int = 1) -> Facade:
        if self.slab_height >= self.floor_height:
            raise ValueError(
                f"slab_height ({self.slab_height}) must be less than floor_height ({self.floor_height})"
            )

        building_segs = utils.explode_curve(self.building_curve)
        facade_type_obj = facade_plan.FacadeTypeRegistry.create_facade_type(
            pattern_type,
            self.pattern_length,
            self.pattern_depth,
            self.pattern_ratio,
            self.building_curve,
        )

        # 기준층(0층) 파사드/슬래브를 한 번만 생성
        base_floor_height = min(self.floor_height, self.building_height)
        facade_height = base_floor_height - self.slab_height
        base_facade = self._generate_base_floor(
            building_segs, facade_type_obj, facade_height
        )

        # 기준층 결과를 층수만큼 복제하여 Z만 이동 배치
        all_glasses, all_walls, all_frames, all_slabs = [], [], [], []
        for floor in range(self.building_floor):
            base_z = floor * self.floor_height
            if base_z >= self.building_height:
                break
            moved = self._move_facade(base_facade, geo.Vector3d(0, 0, base_z))
            all_glasses.extend(moved.glasses)
            all_walls.extend(moved.walls)
            all_frames.extend(moved.frames)
            all_slabs.extend(moved.slabs)

        return Facade(all_glasses, all_walls, all_frames, all_slabs)

    def _generate_floor_slab(self, base_z: float) -> Optional[geo.Brep]:
        """층별 슬래브를 생성하는 메서드"""
        if self.slab_height <= 0:
            return None

        slab_brep = self._create_slab(0)  # Z=0에서 생성
        if slab_brep:
            # base_z 위치로 이동
            slab_brep = utils.move_brep(slab_brep, geo.Vector3d.ZAxis * base_z)
        return slab_brep

    def _generate_floor_facade(
        self, building_segs, facade_type_obj, base_z: float, facade_height: float
    ) -> dict:
        """층별 파사드를 생성하는 메서드"""
        glasses, walls, frames = [], [], []

        facade_z_offset = base_z + (self.slab_height if self.slab_height > 0 else 0)

        for seg in building_segs:
            seg_facade = facade_type_obj.generate(seg, facade_height)
            if not seg_facade:
                continue

            # Z 위치로 이동
            if facade_z_offset > 0:
                seg_facade = self._move_facade(
                    seg_facade, geo.Vector3d.ZAxis * facade_z_offset
                )

            glasses.extend(seg_facade.glasses)
            walls.extend(seg_facade.walls)
            frames.extend(seg_facade.frames)

        return {"glasses": glasses, "walls": walls, "frames": frames}

    def _generate_base_floor(
        self, building_segs, facade_type_obj, facade_height: float
    ) -> Facade:
        """기준층(0층)의 슬래브와 파사드를 한 번 생성하여 반환"""
        # 기준층 슬래브 (Z=0~slab_height)
        slabs: list[geo.Brep] = []
        if self.slab_height > 0:
            slab_brep = self._create_slab(0)
            if slab_brep:
                slabs.append(slab_brep)

        # 기준층 파사드 (슬래브 상단부터 facade_height 만큼)
        glasses, walls, frames = [], [], []
        if facade_height > 0:
            facades = self._generate_floor_facade(
                building_segs, facade_type_obj, 0, facade_height
            )
            glasses.extend(facades["glasses"])
            walls.extend(facades["walls"])
            frames.extend(facades["frames"])

        return Facade(glasses, walls, frames, slabs)

    def _move_facade(self, facade: Facade, vector: geo.Vector3d) -> Facade:
        def _mv(b: geo.Brep) -> geo.Brep:
            return utils.move_brep(b, vector)

        glasses = [_mv(b) for b in facade.glasses if b]
        walls = [_mv(b) for b in facade.walls if b]
        frames = [_mv(b) for b in facade.frames if b]
        slabs = [_mv(b) for b in facade.slabs if b]
        return Facade(glasses, walls, frames, slabs)

    def _offset_slab_curve(self) -> Optional[geo.Curve]:
        """슬래브 커브를 오프셋하는 메서드"""
        if self.slab_offset == 0:
            return self.building_curve

        if self.slab_offset < 0:
            slab_curves = utils.offset_regions_outward(
                self.building_curve, self.slab_offset
            )
        else:
            slab_curves = utils.offset_regions_inward(
                self.building_curve, self.slab_offset
            )

        if not slab_curves:
            print("Failed to offset slab curve.")
            print("slab_offset:", self.slab_offset)
            print(
                "slab_offset 값을 building_curve에 적절한 수치로 지정하거나 building_curve가 적합한 형태인지 확인하세요."
            )
            return None

        return slab_curves[0]

    def _create_slab(self, base_z: float = 0) -> Optional[geo.Brep]:
        """슬래브를 생성하는 메서드 (Z=0에서 생성, 외부에서 이동 처리)"""
        # 슬래브용 커브 생성 (slab_offset만큼 오프셋)
        slab_curve = self.building_curve
        if self.slab_offset != 0:
            slab_curve = self._offset_slab_curve()

        extruded_slab = geo.Extrusion.Create(slab_curve, self.slab_height, True)
        slab_brep_final = extruded_slab.ToBrep()

        # base_z가 0이 아닌 경우에만 이동 (하위 호환성 유지)
        if base_z != 0:
            slab_brep_final = utils.move_brep(
                slab_brep_final, geo.Vector3d.ZAxis * base_z
            )
        return slab_brep_final


if __name__ == "__main__":
    # Grasshopper 컴포넌트 입력이 있을 때만 실행되도록 안전 가드
    _building_brep = globals().get("building_brep", None)
    _floor_height = float(globals().get("floor_height", 3.0))
    _facade_type = int(globals().get("facade_type", 1))
    _pattern_length = float(globals().get("pattern_length", 4.0))
    _pattern_depth = float(globals().get("pattern_depth", 1.0))
    _pattern_ratio = float(globals().get("pattern_ratio", 0.8))
    _pattern_type = int(globals().get("pattern_type", 1))
    _facade_offset = float(globals().get("facade_offset", 0.2))
    _slab_height = float(globals().get("slab_height", 0.0))
    _slab_offset = float(globals().get("slab_offset", 0.0))
    print("MAIN.py - FacadeGenerator parameters:")
    facade_generator = FacadeGenerator(
        _building_brep,
        _floor_height,
        facade_type=_facade_type,
        pattern_length=_pattern_length,
        pattern_depth=_pattern_depth,
        pattern_ratio=_pattern_ratio,
        facade_offset=_facade_offset,
        slab_height=_slab_height,
        slab_offset=_slab_offset,
    )
    facade = facade_generator.generate(_pattern_type)

    glasses = facade.glasses
    walls = facade.walls
    frames = facade.frames
    slabs = facade.slabs
