# -*- coding:utf-8 -*-
"""
BldgDetailBuilder - Merged Version
독립 실행 가능한 파사드 생성기

Dependencies: Rhino.Geometry, ghpythonlib.components
"""

# ==============================================================================
# Imports
# ==============================================================================
try:
    from typing import List, Tuple, Dict, Any, Optional, Union
except ImportError:
    pass

import functools
import math

import Rhino.Geometry as geo  # type: ignore
import scriptcontext as sc  # type: ignore
import Rhino  # type: ignore
import ghpythonlib.components as ghcomp  # type: ignore


# ==============================================================================
# Utils Constants & Utilities
# ==============================================================================
BIGNUM = 10000000
ROUNDING_PRECISION = 6  # 반올림 소수점 자리수

# Tolerances
TOL = 0.01  # 기본 허용 오차
DIST_TOL = 0.01
AREA_TOL = 0.1
OP_TOL = 0.00001
CLIPPER_TOL = 0.0000000001


# ==============================================================================
# Decorators
# ==============================================================================
def convert_io_to_list(func):
    """입력과 출력을 리스트 형태로 일관되게 만들어주는 데코레이터"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, geo.Curve):
                arg = [arg]
            new_args.append(arg)

        result = func(*new_args, **kwargs)
        if isinstance(result, geo.Curve):
            result = [result]

        if hasattr(result, "__dict__"):
            for key, values in result.__dict__.items():
                if isinstance(values, geo.Curve):
                    setattr(result, key, [values])
        return result

    return wrapper


# ==============================================================================
# Core Geometry Utilities
# ==============================================================================
def get_distance_between_points(point_a: geo.Point3d, point_b: geo.Point3d) -> float:
    """두 점 사이의 거리를 계산합니다."""
    return round(point_a.DistanceTo(point_b), ROUNDING_PRECISION)


def get_distance_between_point_and_curve(point: geo.Point3d, curve: geo.Curve) -> float:
    """점과 커브 사이의 최단 거리를 계산합니다."""
    _, param = curve.ClosestPoint(point)
    dist = point.DistanceTo(curve.PointAt(param))
    return round(dist, ROUNDING_PRECISION)


def get_distance_between_curves(curve_a: geo.Curve, curve_b: geo.Curve) -> float:
    """두 커브 사이의 최소 거리를 계산합니다."""
    _, pt_a, pt_b = curve_a.ClosestPoints(curve_b)
    dist = pt_a.DistanceTo(pt_b)
    return round(dist, ROUNDING_PRECISION)


def has_intersection(
    curve_a: geo.Curve,
    curve_b: geo.Curve,
    plane: geo.Plane = geo.Plane.WorldXY,
    tol: float = TOL,
) -> bool:
    """두 커브가 교차하는지 여부를 확인합니다."""
    return geo.Curve.PlanarCurveCollision(curve_a, curve_b, plane, tol)


def get_intersection_points(
    curve_a: geo.Curve, curve_b: geo.Curve, tol: float = TOL
) -> List[geo.Point3d]:
    """두 커브 사이의 교차점을 계산합니다."""
    intersections = geo.Intersect.Intersection.CurveCurve(curve_a, curve_b, tol, tol)
    if not intersections:
        return []
    return [event.PointA for event in intersections if event.IsPointAValid]


def explode_curve(curve: geo.Curve) -> List[geo.Curve]:
    """커브를 분할하여 개별 세그먼트(직선) 리스트로 반환합니다."""
    if not curve:
        return []
    # PolyCurve인 경우, 내부 세그먼트들을 직접 반환
    if isinstance(curve, geo.PolyCurve):
        return list(curve.DuplicateSegments())
    # 일반 커브는 Span 기준으로 분할
    segments = []
    for i in range(curve.SpanCount):
        param_start, param_end = curve.SpanDomain(i)
        pt_start = curve.PointAt(param_start)
        pt_end = curve.PointAt(param_end)
        segments.append(geo.LineCurve(pt_start, pt_end))
    return segments


def get_outside_perp_vec_from_pt(pt: geo.Point3d, region: geo.Curve) -> geo.Vector3d:
    _, param = region.ClosestPoint(pt)
    vec_perp_outer = region.PerpendicularFrameAt(param)[1].XAxis

    if region.ClosedCurveOrientation() != geo.CurveOrientation.Clockwise:
        vec_perp_outer = -vec_perp_outer

    return vec_perp_outer


def get_pts_by_length(
    crv: geo.Curve, length: float, include_start: bool = False
) -> List[geo.Point3d]:
    """커브를 주어진 길이로 나누는 점을 구한다."""
    params = crv.DivideByLength(length, include_start)

    # crv가 length보다 짧은 경우
    if not params:
        return []

    return [crv.PointAt(param) for param in params]


def get_vector_from_pts(pt_a: geo.Point3d, pt_b: geo.Point3d) -> geo.Vector3d:
    """두 점 사이의 벡터를 계산합니다."""
    return geo.Vector3d(pt_b.X - pt_a.X, pt_b.Y - pt_a.Y, pt_b.Z - pt_a.Z)


def get_vertices(curve: geo.Curve) -> List[geo.Point3d]:
    """커브의 모든 정점(Vertex)들을 추출합니다."""
    vertices = [curve.PointAt(curve.SpanDomain(i)[0]) for i in range(curve.SpanCount)]
    if not curve.IsClosed:
        vertices.append(curve.PointAtEnd)
    return vertices


def move_curve(curve: geo.Curve, vector: geo.Vector3d) -> geo.Curve:
    """커브를 주어진 벡터만큼 이동시킨 복사본을 반환합니다."""
    moved_curve = curve.Duplicate()
    moved_curve.Translate(vector)
    return moved_curve


def move_brep(brep: geo.Brep, vector: geo.Vector3d) -> geo.Brep:
    """Brep를 주어진 벡터만큼 이동시킨 복사본을 반환합니다."""
    moved_brep = brep.Duplicate()
    moved_brep.Translate(vector)
    return moved_brep


# ==============================================================================
# Advanced Curve & Region Operations
# ==============================================================================
def get_overlapped_curves(curve_a: geo.Curve, curve_b: geo.Curve) -> List[geo.Curve]:
    """두 커브가 겹치는 구간의 커브들을 반환합니다."""
    if not has_intersection(curve_a, curve_b):
        return []

    intersection_points = get_intersection_points(curve_a, curve_b)
    explode_points = ghcomp.Explode(curve_a, True).vertices + intersection_points
    if not explode_points:
        return []

    params = [ghcomp.CurveClosestPoint(pt, curve_a).parameter for pt in explode_points]
    segments = ghcomp.Shatter(curve_a, params)

    overlapped_segments = [seg for seg in segments if has_intersection(seg, curve_b)]
    if not overlapped_segments:
        return []

    return geo.Curve.JoinCurves(overlapped_segments)


def get_overlapped_length(curve_a: geo.Curve, curve_b: geo.Curve) -> float:
    """두 커브가 겹치는 총 길이를 계산합니다."""
    overlapped_curves = get_overlapped_curves(curve_a, curve_b)
    if not overlapped_curves:
        return 0.0
    return sum(crv.GetLength() for crv in overlapped_curves)


def has_region_intersection(
    region_a: geo.Curve, region_b: geo.Curve, tol: float = TOL
) -> bool:
    """두 닫힌 영역 커브가 교차(겹침 포함)하는지 확인합니다."""
    relationship = geo.Curve.PlanarClosedCurveRelationship(
        region_a, region_b, geo.Plane.WorldXY, tol
    )
    return relationship != geo.RegionContainment.Disjoint


def is_region_inside_region(
    region: geo.Curve, other_region: geo.Curve, tol: float = TOL
) -> bool:
    """'region'이 'other_region' 내부에 완전히 포함되는지 확인합니다."""
    relationship = geo.Curve.PlanarClosedCurveRelationship(
        region, other_region, geo.Plane.WorldXY, tol
    )
    return relationship == geo.RegionContainment.AInsideB


def get_outline_from_closed_brep(brep: geo.Brep, plane: geo.Plane) -> geo.Curve:
    """
    닫힌 폴리서페이스(Brep)를 받아, 주어진 Plane 기준으로 Contour를 생성하고,
    결과 커브들 중 Z값이 가장 낮은 커브를 반환합니다.
    brep가 닫힌 Brep가 아니면 TypeError를 발생시킵니다.
    """
    if not isinstance(brep, geo.Brep) or not brep.IsSolid:
        raise TypeError("입력은 닫힌 Brep(폴리서페이스)만 허용됩니다.")
    bbox = brep.GetBoundingBox(True)
    contour_start = geo.Point3d(0, 0, bbox.Min.Z)
    contour_end = geo.Point3d(0, 0, bbox.Max.Z)
    curves = geo.Brep.CreateContourCurves(
        brep, contour_start, contour_end, (bbox.Max.Z - bbox.Min.Z)
    )

    if not curves or len(curves) == 0:
        return None

    # Z값이 가장 낮은 커브 선택 (평균 Z값 기준)
    def avg_z(curve):
        return curve.PointAtStart.Z

    return min(curves, key=avg_z)


class Offset:
    """RhinoCommon 기반 오프셋 유틸리티 (Clipper 미사용)"""

    class _PolylineOffsetResult:
        def __init__(self):
            self.contour: Optional[List[geo.Curve]] = None
            self.holes: Optional[List[geo.Curve]] = None

    @convert_io_to_list
    def polyline_offset(
        self, curves: List[geo.Curve], dists: Union[float, List[float]], **kwargs
    ) -> _PolylineOffsetResult:
        if not curves:
            raise ValueError("No Curves to offset")

        # 옵션 처리 (필요시 확장 가능)
        tol = kwargs.get("tol", Rhino.RhinoMath.ZeroTolerance)
        plane = kwargs.get("plane", geo.Plane.WorldXY)

        # 거리 목록 정규화
        if isinstance(dists, (int, float)):
            dist_list = [float(dists)] * len(curves)
        else:
            dist_list = [float(d) for d in dists]
            if len(dist_list) != len(curves):
                raise ValueError(
                    "Length of dists must match curves or be a single number"
                )

        outward_all: List[geo.Curve] = []
        inward_all: List[geo.Curve] = []

        for crv, dist in zip(curves, dist_list):
            if not crv or not crv.IsClosed:
                # 열린 커브는 Offset 결과 해석이 애매하므로 스킵
                # 필요 시 open curve 지원 로직 추가 가능
                continue

            # 커브의 자체 평면을 우선 사용 (불가하면 입력 plane)
            try:
                ok, crv_plane = crv.TryGetPlane()
            except Exception:
                ok, crv_plane = False, None
            plane_used = crv_plane if ok and crv_plane else plane

            # 기준 면적 계산 (원래 커브 면적)
            orig_breps = geo.Brep.CreatePlanarBreps(crv, tol)
            orig_area = 0.0
            if orig_breps:
                for b in orig_breps:
                    amp = geo.AreaMassProperties.Compute(b)
                    if amp:
                        orig_area += amp.Area

            # 양/음 오프셋 모두 계산 후 면적 비교로 outward/inward를 결정
            d = abs(dist)

            def do_offset(distance: float):
                try:
                    return crv.Offset(
                        plane_used, distance, tol, geo.CurveOffsetCornerStyle.Sharp
                    )
                except TypeError:
                    return crv.Offset(plane_used, distance, tol)

            pos = do_offset(+d)
            neg = do_offset(-d)

            def total_area(curves_list: Optional[List[geo.Curve]]) -> float:
                if not curves_list:
                    return -1.0
                area_sum = 0.0
                for c in curves_list:
                    breps = geo.Brep.CreatePlanarBreps(c, tol)
                    if not breps:
                        continue
                    for b in breps:
                        amp = geo.AreaMassProperties.Compute(b)
                        if amp:
                            area_sum += amp.Area
                return area_sum

            area_pos = total_area(pos)
            area_neg = total_area(neg)

            # 원래 면적보다 큰 쪽을 outward로 채택 (둘 다 유효하지 않으면 스킵)
            chosen_out, chosen_in = None, None
            if area_pos <= 0 and area_neg <= 0:
                continue
            if orig_area > 0:
                # 원 면적 대비 증가한 쪽이 outward
                if area_pos > area_neg:
                    chosen_out, chosen_in = pos, neg
                else:
                    chosen_out, chosen_in = neg, pos
            else:
                # 원 면적을 구할 수 없으면 더 큰 면적을 outward로 가정
                if area_pos >= area_neg:
                    chosen_out, chosen_in = pos, neg
                else:
                    chosen_out, chosen_in = neg, pos

            if chosen_out:
                outward_all.extend(list(chosen_out))
            if chosen_in:
                inward_all.extend(list(chosen_in))

        offset_result = Offset._PolylineOffsetResult()
        offset_result.contour = outward_all
        offset_result.holes = inward_all
        return offset_result


def offset_regions_inward(
    regions: Union[geo.Curve, List[geo.Curve]], dist: float, **kwargs
) -> List[geo.Curve]:
    """닫힌 영역(들)을 안쪽으로 오프셋합니다."""
    if not dist:
        return regions if isinstance(regions, list) else [regions]
    res = Offset().polyline_offset(
        regions if isinstance(regions, list) else [regions], dist, **kwargs
    )
    return res.holes or []


def offset_regions_outward(
    regions: Union[geo.Curve, List[geo.Curve]], dist: float, **kwargs
) -> List[geo.Curve]:
    """닫힌 영역(들)을 바깥쪽으로 오프셋합니다."""
    if not dist:
        return regions if isinstance(regions, list) else [regions]
    res = Offset().polyline_offset(
        regions if isinstance(regions, list) else [regions], dist, **kwargs
    )
    return res.contour or []


# ==============================================================================
# Main Facade Generation Classes
# ==============================================================================

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
        self.building_curve = get_outline_from_closed_brep(
            building_brep, geo.Plane.WorldXY
        )
        self.building_curve = offset_regions_outward(
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
        building_segs = explode_curve(self.building_curve)

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
            return move_brep(b, vector)

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
        if pattern_type == 4:
            return self.generate_facade_type_4(seg, extrude_height)
        if pattern_type == 5:
            return self.generate_facade_type_5(seg, extrude_height)
        if pattern_type == 6:
            return self.generate_facade_type_6(seg, extrude_height)
        if pattern_type == 7:
            return self.generate_facade_type_7(seg, extrude_height)
        if pattern_type == 8:
            return self.generate_facade_type_8(seg, extrude_height)
        if pattern_type == 9:
            return self.generate_facade_type_9(seg, extrude_height)
        if pattern_type == 10:
            return self.generate_facade_type_10(seg, extrude_height)

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
        pts_from_seg = get_pts_by_length(
            seg, self.pattern_length, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        for pt, next_pt in zip(pts_from_seg, pts_from_seg[1:]):
            vector = get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * (self.pattern_length * self.pattern_ratio)

            out_vec: Optional[geo.Vector3d] = None
            if offset_partition_outward or add_frame_outward:
                out_vec = get_outside_perp_vec_from_pt(
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

    # 타입 4: 지그재그 패턴 파사드 - 교대로 안쪽/바깥쪽으로 들어가는 패턴
    def generate_facade_type_4(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """지그재그 패턴 파사드 생성 - 교대로 안쪽/바깥쪽으로 오프셋"""
        pts_from_seg = get_pts_by_length(
            seg, self.pattern_length, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        for i, (pt, next_pt) in enumerate(zip(pts_from_seg, pts_from_seg[1:])):
            vector = get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * (self.pattern_length * self.pattern_ratio)

            # 교대로 안쪽/바깥쪽으로 offset
            out_vec = get_outside_perp_vec_from_pt(div_pt, self.building_curve)
            if out_vec is not None:
                # 홀수/짝수에 따라 방향 결정
                direction = 1 if i % 2 == 0 else -1
                zigzag_pt = div_pt + out_vec * (self.pattern_depth * direction)

                # 지그재그 프레임 추가 (바깥쪽 방향에서만)
                if direction > 0:  # 바깥쪽으로 나가는 지점에서만
                    frame_end = div_pt + out_vec * (
                        self.pattern_depth * direction * 0.6
                    )
                    frame_segs.append(geo.LineCurve(div_pt, frame_end))

                glass_segs.append(geo.LineCurve(pt, zigzag_pt))
                wall_segs.append(geo.LineCurve(zigzag_pt, next_pt))
            else:
                glass_segs.append(geo.LineCurve(pt, div_pt))
                wall_segs.append(geo.LineCurve(div_pt, next_pt))

        # 마지막 점 처리
        if len(pts_from_seg) >= 2:
            wall_segs.append(geo.LineCurve(pts_from_seg[-1], seg.PointAtEnd))

        return self._extrude_to_facade(glass_segs, wall_segs, frame_segs, extrude_height)

    # 타입 5: 웨이브 패턴 파사드 - 사인파 형태로 깊이 변화
    def generate_facade_type_5(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """웨이브 패턴 파사드 생성 - 사인파 형태로 깊이 변화"""
        pts_from_seg = get_pts_by_length(
            seg, self.pattern_length, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        for i, (pt, next_pt) in enumerate(zip(pts_from_seg, pts_from_seg[1:])):
            vector = get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * (self.pattern_length * self.pattern_ratio)

            # 사인파를 이용한 웨이브 패턴
            out_vec = get_outside_perp_vec_from_pt(div_pt, self.building_curve)
            if out_vec is not None:
                wave_factor = math.sin(i * math.pi / 2)  # 0, 1, 0, -1, 0, 1, ... 패턴
                wave_pt = div_pt + out_vec * (self.pattern_depth * wave_factor)

                # 웨이브 프레임 추가 (정점과 골에서만)
                if abs(wave_factor) > 0.7:  # 강한 웨이브 지점에서만
                    frame_depth = self.pattern_depth * abs(wave_factor) * 0.5
                    frame_end = div_pt + out_vec * frame_depth
                    frame_segs.append(geo.LineCurve(div_pt, frame_end))

                glass_segs.append(geo.LineCurve(pt, wave_pt))
                wall_segs.append(geo.LineCurve(wave_pt, next_pt))
            else:
                glass_segs.append(geo.LineCurve(pt, div_pt))
                wall_segs.append(geo.LineCurve(div_pt, next_pt))

        # 마지막 점 처리
        if len(pts_from_seg) >= 2:
            wall_segs.append(geo.LineCurve(pts_from_seg[-1], seg.PointAtEnd))

        return self._extrude_to_facade(glass_segs, wall_segs, frame_segs, extrude_height)

    # 타입 6: 피라미드 패턴 파사드 - 중앙으로 갈수록 더 깊게 들어가는 패턴
    def generate_facade_type_6(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """피라미드 패턴 파사드 생성 - 중앙으로 갈수록 더 깊게"""
        pts_from_seg = get_pts_by_length(
            seg, self.pattern_length, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        total_segments = len(pts_from_seg) - 1

        for i, (pt, next_pt) in enumerate(zip(pts_from_seg, pts_from_seg[1:])):
            vector = get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * (self.pattern_length * self.pattern_ratio)

            # 피라미드 형태: 중앙에 가까울수록 더 깊게
            out_vec = get_outside_perp_vec_from_pt(div_pt, self.building_curve)
            if out_vec is not None:
                # 중앙으로부터의 거리에 따른 깊이 계산
                center_distance = abs(i - total_segments / 2) / (total_segments / 2)
                pyramid_factor = 1.0 - center_distance  # 중앙이 1.0, 끝이 0.0
                pyramid_pt = div_pt + out_vec * (self.pattern_depth * pyramid_factor)

                # 프레임도 추가 (피라미드 형태를 강조)
                if pyramid_factor > 0.3:  # 일정 깊이 이상에서만 프레임 생성
                    frame_end = div_pt + out_vec * (
                        self.pattern_depth * pyramid_factor * 0.5
                    )
                    frame_segs.append(geo.LineCurve(div_pt, frame_end))

                glass_segs.append(geo.LineCurve(pt, pyramid_pt))
                wall_segs.append(geo.LineCurve(pyramid_pt, next_pt))
            else:
                glass_segs.append(geo.LineCurve(pt, div_pt))
                wall_segs.append(geo.LineCurve(div_pt, next_pt))

        # 마지막 점 처리
        if len(pts_from_seg) >= 2:
            wall_segs.append(geo.LineCurve(pts_from_seg[-1], seg.PointAtEnd))

        return self._extrude_to_facade(
            glass_segs, wall_segs, frame_segs, extrude_height
        )

    # 타입 7: 헥사곤 패턴 - 육각형 모양으로 안쪽/바깥쪽 교대 배치
    def generate_facade_type_7(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """헥사곤 패턴 파사드 생성 - 육각형 리듬으로 다이나믹한 패턴"""
        pts_from_seg = get_pts_by_length(
            seg, self.pattern_length * 0.5, include_start=True  # 더 세밀한 분할
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        for i, (pt, next_pt) in enumerate(zip(pts_from_seg, pts_from_seg[1:])):
            vector = get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * ((self.pattern_length * 0.5) * self.pattern_ratio)

            # 헥사곤 패턴 (6개 주기로 반복)
            out_vec = get_outside_perp_vec_from_pt(div_pt, self.building_curve)
            if out_vec is not None:
                hex_angle = (i % 6) * math.pi / 3  # 0, 60, 120, 180, 240, 300도
                hex_factor = math.cos(hex_angle) * 0.8 + 0.2  # -0.6 ~ 1.0 범위로 조정
                hex_pt = div_pt + out_vec * (self.pattern_depth * hex_factor)

                # 프레임 추가 (특정 각도에서만)
                if abs(hex_factor) > 0.5:
                    frame_end = div_pt + out_vec * (
                        self.pattern_depth * hex_factor * 0.3
                    )
                    frame_segs.append(geo.LineCurve(div_pt, frame_end))

                glass_segs.append(geo.LineCurve(pt, hex_pt))
                wall_segs.append(geo.LineCurve(hex_pt, next_pt))
            else:
                glass_segs.append(geo.LineCurve(pt, div_pt))
                wall_segs.append(geo.LineCurve(div_pt, next_pt))

        if len(pts_from_seg) >= 2:
            wall_segs.append(geo.LineCurve(pts_from_seg[-1], seg.PointAtEnd))

        return self._extrude_to_facade(
            glass_segs, wall_segs, frame_segs, extrude_height
        )

    # 타입 8: 스파이럴 패턴 - 나선형으로 깊이가 변화하는 패턴
    def generate_facade_type_8(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """스파이럴 패턴 파사드 생성 - 나선형 깊이 변화로 매우 다이나믹"""
        pts_from_seg = get_pts_by_length(
            seg, self.pattern_length * 0.8, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        for i, (pt, next_pt) in enumerate(zip(pts_from_seg, pts_from_seg[1:])):
            vector = get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * ((self.pattern_length * 0.8) * self.pattern_ratio)

            # 스파이럴 패턴 (계속 증가하는 나선)
            out_vec = get_outside_perp_vec_from_pt(div_pt, self.building_curve)
            if out_vec is not None:
                spiral_angle = i * math.pi / 4  # 45도씩 회전
                spiral_radius = (i % 8 + 1) / 8.0  # 1/8부터 1까지 순환
                spiral_factor = math.sin(spiral_angle) * spiral_radius
                spiral_pt = div_pt + out_vec * (self.pattern_depth * spiral_factor)

                # 동적 프레임 (스파이럴 강도에 따라)
                if abs(spiral_factor) > 0.4:
                    frame_depth = self.pattern_depth * abs(spiral_factor) * 0.6
                    frame_end = div_pt + out_vec * frame_depth
                    frame_segs.append(geo.LineCurve(div_pt, frame_end))

                glass_segs.append(geo.LineCurve(pt, spiral_pt))
                wall_segs.append(geo.LineCurve(spiral_pt, next_pt))
            else:
                glass_segs.append(geo.LineCurve(pt, div_pt))
                wall_segs.append(geo.LineCurve(div_pt, next_pt))

        if len(pts_from_seg) >= 2:
            wall_segs.append(geo.LineCurve(pts_from_seg[-1], seg.PointAtEnd))

        return self._extrude_to_facade(
            glass_segs, wall_segs, frame_segs, extrude_height
        )

    # 타입 9: 랜덤 노이즈 패턴 - 의사 랜덤으로 불규칙한 깊이
    def generate_facade_type_9(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """랜덤 노이즈 패턴 파사드 생성 - 불규칙하고 유기적인 형태"""
        pts_from_seg = get_pts_by_length(
            seg, self.pattern_length * 0.6, include_start=True
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        for i, (pt, next_pt) in enumerate(zip(pts_from_seg, pts_from_seg[1:])):
            vector = get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * ((self.pattern_length * 0.6) * self.pattern_ratio)

            # 의사 랜덤 노이즈 (시드 기반으로 일관성 유지)
            out_vec = get_outside_perp_vec_from_pt(div_pt, self.building_curve)
            if out_vec is not None:
                # 여러 주파수 노이즈 합성
                noise1 = math.sin(i * 0.7) * 0.5
                noise2 = math.cos(i * 1.3) * 0.3
                noise3 = math.sin(i * 2.1) * 0.2
                noise_factor = noise1 + noise2 + noise3

                noise_pt = div_pt + out_vec * (self.pattern_depth * noise_factor)

                # 강한 노이즈 지점에서 프레임 생성
                if abs(noise_factor) > 0.6:
                    frame_end = div_pt + out_vec * (
                        self.pattern_depth * noise_factor * 0.4
                    )
                    frame_segs.append(geo.LineCurve(div_pt, frame_end))

                glass_segs.append(geo.LineCurve(pt, noise_pt))
                wall_segs.append(geo.LineCurve(noise_pt, next_pt))
            else:
                glass_segs.append(geo.LineCurve(pt, div_pt))
                wall_segs.append(geo.LineCurve(div_pt, next_pt))

        if len(pts_from_seg) >= 2:
            wall_segs.append(geo.LineCurve(pts_from_seg[-1], seg.PointAtEnd))

        return self._extrude_to_facade(
            glass_segs, wall_segs, frame_segs, extrude_height
        )

    # 타입 10: 복합 패턴 - 여러 패턴이 결합된 최고 다이나믹 패턴
    def generate_facade_type_10(
        self, seg: geo.Curve, extrude_height: float
    ) -> Optional[Facade]:
        """복합 패턴 파사드 생성 - 지그재그 + 웨이브 + 피라미드 + 노이즈 결합"""
        pts_from_seg = get_pts_by_length(
            seg, self.pattern_length * 0.4, include_start=True  # 가장 세밀한 분할
        )
        if not pts_from_seg or len(pts_from_seg) < 2:
            return None

        glass_segs: list[geo.Curve] = []
        wall_segs: list[geo.Curve] = []
        frame_segs: list[geo.Curve] = []

        total_segments = len(pts_from_seg) - 1

        for i, (pt, next_pt) in enumerate(zip(pts_from_seg, pts_from_seg[1:])):
            vector = get_vector_from_pts(pt, next_pt)
            if hasattr(vector, "IsZero") and vector.IsZero:
                continue
            vector.Unitize()

            div_pt = pt + vector * ((self.pattern_length * 0.4) * self.pattern_ratio)

            out_vec = get_outside_perp_vec_from_pt(div_pt, self.building_curve)
            if out_vec is not None:
                # 1. 지그재그 성분
                zigzag_factor = 1 if i % 2 == 0 else -1

                # 2. 웨이브 성분
                wave_factor = math.sin(i * math.pi / 3) * 0.7

                # 3. 피라미드 성분
                center_distance = (
                    abs(i - total_segments / 2) / (total_segments / 2)
                    if total_segments > 0
                    else 0
                )
                pyramid_factor = (1.0 - center_distance) * 0.6

                # 4. 노이즈 성분
                noise_factor = math.sin(i * 0.9) * math.cos(i * 1.7) * 0.4

                # 모든 성분 결합
                combined_factor = (
                    zigzag_factor * 0.3
                    + wave_factor * 0.3
                    + pyramid_factor * 0.2
                    + noise_factor * 0.2
                )

                complex_pt = div_pt + out_vec * (self.pattern_depth * combined_factor)

                # 복잡한 프레임 시스템
                if abs(combined_factor) > 0.5:
                    # 메인 프레임
                    frame_end = div_pt + out_vec * (
                        self.pattern_depth * combined_factor * 0.5
                    )
                    frame_segs.append(geo.LineCurve(div_pt, frame_end))

                    # 보조 프레임 (강한 지점에서만)
                    if abs(combined_factor) > 0.8:
                        side_frame_end = div_pt + out_vec * (
                            self.pattern_depth * combined_factor * 0.3
                        )
                        frame_segs.append(geo.LineCurve(div_pt, side_frame_end))

                glass_segs.append(geo.LineCurve(pt, complex_pt))
                wall_segs.append(geo.LineCurve(complex_pt, next_pt))
            else:
                glass_segs.append(geo.LineCurve(pt, div_pt))
                wall_segs.append(geo.LineCurve(div_pt, next_pt))

        if len(pts_from_seg) >= 2:
            wall_segs.append(geo.LineCurve(pts_from_seg[-1], seg.PointAtEnd))

        return self._extrude_to_facade(
            glass_segs, wall_segs, frame_segs, extrude_height
        )


# ==============================================================================
# Main Execution (Grasshopper Compatible)
# ==============================================================================
if __name__ == "__main__":
    # Grasshopper 컴포넌트 입력이 있을 때만 실행되도록 안전 가드
    _building_brep = globals().get("building_brep", None)
    _floor_height = globals().get("floor_height", None)
    _facade_type = globals().get("facade_type", None)
    _pattern_length = globals().get("pattern_length", None)
    _pattern_depth = globals().get("pattern_depth", None)
    _pattern_ratio = globals().get("pattern_ratio", None)
    _pattern_type = globals().get("pattern_type", None)
    _facade_offset = globals().get("facade_offset", None)

    if _building_brep is not None and _floor_height is not None:
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
    else:
        # 입력이 없을 때 기본값 설정
        glasses = []
        walls = []
        frames = []
