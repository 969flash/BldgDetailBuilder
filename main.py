# -*- coding:utf-8 -*-
# r: clipper
try:
    from typing import List, Tuple, Dict, Any, Optional
except ImportError:
    pass

import Rhino.Geometry as geo  # type: ignore
import scriptcontext as sc  # type: ignore
import Rhino  # type: ignore
import ghpythonlib.components as ghcomp  # type: ignore
import utils
import importlib

# 모듈 새로고침
importlib.reload(utils)
####
# input : 3D Building  Brep
# output : Building Facade Breps
####


class Facade:
    def __init__(
        self,
        glasses: list[geo.Brep],
        walls: list[geo.Brep],
        frames: list[geo.Brep] = None,
    ) -> None:
        self.glasses = glasses
        self.walls = walls
        self.frames = frames if frames is not None else []


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
    ):
        self.building_brep = building_brep
        self.building_curve = utils.get_outline_from_closed_brep(
            building_brep, geo.Plane.WorldXY
        )
        self.building_curve = utils.offset_regions_outward(
            self.building_curve, facade_offset
        )
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

    def _get_building_height(self) -> float:
        # 건물의 높이를 계산하는 로직을 구현합니다.
        bbox = self.building_brep.GetBoundingBox(True)
        return bbox.Max.Z - bbox.Min.Z

    def generate_facade(self, pattern_type: Optional[int] = None) -> Facade:
        # 하나의 건물 메스에 대해, 모든 세그먼트 × 모든 층의 결과를 하나의 Facade로 집계
        building_segs = utils.explode_curve(self.building_curve)

        all_glasses: list[geo.Brep] = []
        all_walls: list[geo.Brep] = []
        all_frames: list[geo.Brep] = []

        for floor in range(self.building_floor):
            base_z = floor * self.floor_height
            if base_z >= self.building_height:
                break
            # 마지막 층은 남은 높이만큼만 생성
            extrude_h = min(self.floor_height, self.building_height - base_z)

            for seg in building_segs:
                seg_facade = self._dispatch_generate(seg, extrude_h, pattern_type)
                if not seg_facade:
                    continue
                if base_z > 0:
                    seg_facade = self._move_facade(
                        seg_facade, geo.Vector3d.ZAxis * base_z
                    )
                all_glasses.extend(seg_facade.glasses)
                all_walls.extend(seg_facade.walls)
                all_frames.extend(seg_facade.frames)

        return Facade(all_glasses, all_walls, all_frames)

    def _move_facade(self, facade: Facade, vector: geo.Vector3d) -> Facade:
        def _mv(b: geo.Brep) -> geo.Brep:
            return utils.move_brep(b, vector)

        glasses = [_mv(b) for b in facade.glasses if b]
        walls = [_mv(b) for b in facade.walls if b]
        frames = [_mv(b) for b in facade.frames if b]
        return Facade(glasses, walls, frames)

    def _dispatch_generate(
        self, seg: geo.Curve, extrude_height: float, pattern_type: Optional[int] = None
    ) -> Optional[Facade]:
        """facade_type에 따라 타입별 생성 함수로 라우팅"""
        if pattern_type == 1:
            return self.generate_facade_type_1(seg, extrude_height)
        if pattern_type == 2:
            return self.generate_facade_type_2(seg, extrude_height)
        if pattern_type == 3:
            return self.generate_facade_type_3(seg, extrude_height)

        return self.generate_facade_type_1(seg, extrude_height)

    def _generate_facades_from_seg(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """패턴 파라미터 기반의 기본(타입1) 파사드 생성"""
        # 패턴 파라미터 사용
        pts_from_seg = utils.get_pts_by_length(
            seg, self.pattern_length, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []

        # 인접한 점 쌍으로 안전하게 순회
        for pt, next_pt in zip(pts_from_seg, pts_from_seg[1:]):
            vector = utils.get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()
            mid_pt = pt + vector * (self.pattern_length * self.pattern_ratio)
            out_vector = utils.get_outside_perp_vec_from_pt(mid_pt, self.building_curve)
            mid_pt += out_vector * self.pattern_depth

            glass_segs.append(geo.LineCurve(pt, mid_pt))
            wall_segs.append(geo.LineCurve(mid_pt, next_pt))

        def _ext_to_brep(line: geo.Curve) -> Optional[geo.Brep]:
            ext = geo.Extrusion.Create(line, extrude_height, False)
            return ext.ToBrep() if ext else None

        glass_breps = [b for b in (_ext_to_brep(seg) for seg in glass_segs) if b]
        wall_breps = [b for b in (_ext_to_brep(seg) for seg in wall_segs) if b]
        if not glass_breps and not wall_breps:
            return None
        return Facade(glass_breps, wall_breps)

    def generate_facade_type_1(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        return self._generate_facades_from_seg(seg, extrude_height)

    # 타입 2: 직선형 파사드
    def generate_facade_type_2(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """직선형 파사드 생성, 패턴 비율에 따라 glass 와 wall 반복"""
        pts_from_seg = utils.get_pts_by_length(
            seg, self.pattern_length, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []

        # 인접한 점 쌍으로 안전하게 순회
        for pt, next_pt in zip(pts_from_seg, pts_from_seg[1:]):
            vector = utils.get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()
            mid_pt = pt + vector * (self.pattern_length * self.pattern_ratio)
            glass_segs.append(geo.LineCurve(pt, mid_pt))
            wall_segs.append(geo.LineCurve(mid_pt, next_pt))

        def _ext_to_brep(line: geo.Curve) -> Optional[geo.Brep]:
            ext = geo.Extrusion.Create(line, extrude_height, False)
            return ext.ToBrep() if ext else None

        glass_breps = [b for b in (_ext_to_brep(seg) for seg in glass_segs) if b]
        wall_breps = [b for b in (_ext_to_brep(seg) for seg in wall_segs) if b]
        if not glass_breps and not wall_breps:
            return None
        return Facade(glass_breps, wall_breps)

    # 타입 3: (임시) 타입 1과 동일 동작. 이후 확장 가능
    def generate_facade_type_3(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        # 패턴 파라미터 사용
        pts_from_seg = utils.get_pts_by_length(
            seg, self.pattern_length, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        # 인접한 점 쌍으로 안전하게 순회
        for pt, next_pt in zip(pts_from_seg, pts_from_seg[1:]):
            vector = utils.get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()
            mid_pt = pt + vector * (self.pattern_length * self.pattern_ratio)
            out_vector = utils.get_outside_perp_vec_from_pt(mid_pt, self.building_curve)
            out_mid_pt = mid_pt + out_vector * self.pattern_depth

            frame_segs.append(geo.LineCurve(mid_pt, out_mid_pt))
            glass_segs.append(geo.LineCurve(pt, mid_pt))
            wall_segs.append(geo.LineCurve(mid_pt, next_pt))

        def _ext_to_brep(line: geo.Curve) -> Optional[geo.Brep]:
            ext = geo.Extrusion.Create(line, extrude_height, False)
            return ext.ToBrep() if ext else None

        glass_breps = [b for b in (_ext_to_brep(seg) for seg in glass_segs) if b]
        wall_breps = [b for b in (_ext_to_brep(seg) for seg in wall_segs) if b]
        frame_breps = [b for b in (_ext_to_brep(seg) for seg in frame_segs) if b]
        if not glass_breps and not wall_breps:
            return None
        return Facade(glass_breps, wall_breps, frame_breps)


# Grasshopper 컴포넌트 입력이 있을 때만 실행되도록 안전 가드
_building_brep = globals().get("building_brep", None)
_floor_height = globals().get("floor_height", None)
_facade_type = globals().get("facade_type", None)
_pattern_length = globals().get("pattern_length", None)
_pattern_depth = globals().get("pattern_depth", None)
_pattern_ratio = globals().get("pattern_ratio", None)
_pattern_type = globals().get("pattern_type", None)
_facade_offset = globals().get("facade_offset", None)

facade_generator = FacadeGenerator(
    _building_brep,
    float(_floor_height),
    facade_type=int(_facade_type) if _facade_type is not None else 1,
    pattern_length=float(_pattern_length) if _pattern_length is not None else 4.0,
    pattern_depth=float(_pattern_depth) if _pattern_depth is not None else 1.0,
    pattern_ratio=float(_pattern_ratio) if _pattern_ratio is not None else 0.8,
    facade_offset=float(_facade_offset) if _facade_offset is not None else 0.2,
)
facade = facade_generator.generate_facade(_pattern_type)

glasses = facade.glasses
walls = facade.walls
frames = facade.frames
