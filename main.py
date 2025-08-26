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

    # 내부 유틸: 패턴화된 분할점을 기준으로 glass/wall(/frame) 커브 세트를 생성
    def _segs_by_pattern(
        self,
        seg: geo.Curve,
        *,
        offset_partition_outward: bool,
        add_frame_outward: bool,
    ) -> Optional[tuple[list[geo.Curve], list[geo.Curve], list[geo.Curve]]]:
        """주어진 세그먼트를 패턴 길이/비율로 분할하고 역할별 커브 리스트를 생성한다.

        - offset_partition_outward: 분할점 자체를 외측으로 이동해 glass/wall 경계에 깊이를 반영할지 여부 (type1)
        - add_frame_outward: 분할점에서 외측으로 프레임 커브를 추가할지 여부 (type3)
        """
        pts_from_seg = utils.get_pts_by_length(
            seg, self.pattern_length, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        for pt, next_pt in zip(pts_from_seg, pts_from_seg[1:]):
            vector = utils.get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * (self.pattern_length * self.pattern_ratio)

            out_vec: Optional[geo.Vector3d] = None
            if offset_partition_outward or add_frame_outward:
                out_vec = utils.get_outside_perp_vec_from_pt(
                    div_pt, self.building_curve
                )

            # 프레임 커브 생성 (type3)
            if add_frame_outward and out_vec is not None:
                out_mid = div_pt + out_vec * self.pattern_depth
                frame_segs.append(geo.LineCurve(div_pt, out_mid))

            # 분할점을 외측으로 밀어 경계를 형성할지 (type1)
            partition_pt = (
                div_pt + (out_vec * self.pattern_depth)
                if (offset_partition_outward and out_vec is not None)
                else div_pt
            )

            glass_segs.append(geo.LineCurve(pt, partition_pt))
            wall_segs.append(geo.LineCurve(partition_pt, next_pt))

        # 마지막 점과 세그먼트 끝점 연결은 wall 처리
        if len(pts_from_seg) >= 2:
            last_pt = pts_from_seg[-1]
            end_pt = seg.PointAtEnd
            wall_segs.append(geo.LineCurve(last_pt, end_pt))

        return glass_segs, wall_segs, frame_segs

    # 내부 유틸: 커브 세트를 층 높이만큼 압출하여 Facade로 변환
    def _extrude_to_facade(
        self,
        glass_segs: list[geo.Curve],
        wall_segs: list[geo.Curve],
        frame_segs: list[geo.Curve],
        extrude_height: float,
    ) -> Optional[Facade]:
        def _ext_to_brep(line: geo.Curve) -> Optional[geo.Brep]:
            ext = geo.Extrusion.Create(line, extrude_height, False)
            return ext.ToBrep() if ext else None

        glass_breps = [b for b in (_ext_to_brep(c) for c in glass_segs) if b]
        wall_breps = [b for b in (_ext_to_brep(c) for c in wall_segs) if b]
        frame_breps = [b for b in (_ext_to_brep(c) for c in frame_segs) if b]
        if not glass_breps and not wall_breps and not frame_breps:
            return None
        return Facade(glass_breps, wall_breps, frame_breps)

    def generate_facade_type_1(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """패턴 파라미터 기반의 기본(타입1) 파사드 생성"""
        segs = self._segs_by_pattern(
            seg,
            offset_partition_outward=True,  # 분할점 자체를 외측으로 이동해 경계 형성
            add_frame_outward=False,  # 프레임 없음
        )
        if not segs:
            return None
        glass_segs, wall_segs, frame_segs = segs
        return self._extrude_to_facade(
            glass_segs, wall_segs, frame_segs, extrude_height
        )

    # 타입 2: 직선형 파사드
    def generate_facade_type_2(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """직선형 파사드 생성, 패턴 비율에 따라 glass 와 wall 반복"""
        segs = self._segs_by_pattern(
            seg,
            offset_partition_outward=False,  # 분할점 외측 이동 없음
            add_frame_outward=False,  # 프레임 없음
        )
        if not segs:
            return None
        glass_segs, wall_segs, frame_segs = segs
        return self._extrude_to_facade(
            glass_segs, wall_segs, frame_segs, extrude_height
        )

    # 타입 3: (임시) 타입 1과 동일 동작. 이후 확장 가능
    def generate_facade_type_3(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        segs = self._segs_by_pattern(
            seg,
            offset_partition_outward=False,  # 분할점 외측 이동 없음
            add_frame_outward=True,  # 프레임 생성
        )
        if not segs:
            return None
        glass_segs, wall_segs, frame_segs = segs
        return self._extrude_to_facade(
            glass_segs, wall_segs, frame_segs, extrude_height
        )


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
