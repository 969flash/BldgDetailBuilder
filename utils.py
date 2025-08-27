# -*- coding:utf-8 -*-
# utils_geometry.py

# ==============================================================================
# Imports
# ==============================================================================
import functools
from typing import List, Tuple, Any, Optional, Union

# Rhino Libraries
import Rhino
import Rhino.Geometry as geo
import ghpythonlib.components as ghcomp


# ==============================================================================
# Constants
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
