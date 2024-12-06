import ghpythonlib.components as ghcomp
import Rhino.Geometry as geo
import Rhino
import scriptcontext as sc
import random
import rhinoscriptsyntax as rs


class Floor:
    def __init__(self, brep):
        self.height = 0.0
        self.brep = brep
        self.faces = None
        self.floor = None
        self.ceiling = None
        self.glass = None
        self.glass_frames = None
        self.core = None
        self.slab = None
        self._initialize()

    def _initialize(self):
        def get_max_z(brep):
            return max(pt.Location.Z for pt in brep.Vertices)

        def get_min_z(brep):
            return min(pt.Location.Z for pt in brep.Vertices)

        self.breps = ghcomp.DeconstructBrep(self.brep)[0]
        self.floor = min(self.breps, key=lambda brep: get_max_z(brep))
        self.ceiling = max(self.breps, key=lambda brep: get_min_z(brep))
        self.glass = [
            face for face in self.breps if face != self.floor and face != self.ceiling
        ]
        self.height = (
            self.ceiling.Vertices[0].Location.Z - self.floor.Vertices[0].Location.Z
        )

    def set_glass_frames(self, distance, frame_distance):
        if self.glass is None:
            return
        glass_frames = []
        edges = self.floor.DuplicateEdgeCurves()
        for edge in edges:
            params = edge.DivideByLength(distance, True)
            if not params:
                continue
            for i in range(len(params) - 1):
                segment = edge.Trim(params[i], params[i + 1])

                if segment:
                    frame_curve = segment.DuplicateCurve()
                    frame_curve = trim_crv_from_length(frame_curve, frame_distance)

                    if not frame_curve:
                        continue
                    frame_surface = geo.Extrusion.Create(
                        frame_curve, self.height, False
                    )
                    glass_frames.append(frame_surface)

        self.glass_frames = glass_frames

    def _get_core(self, offset_distance, reverse=False, height=None):
        if self.ceiling is None:
            return
        if height is None:
            height = self.height
        ceiling_edges = self.ceiling.DuplicateEdgeCurves()
        ceiling_edge = ghcomp.JoinCurves(ceiling_edges, True)
        core_edge = ghcomp.OffsetCurve(ceiling_edge, -offset_distance, corners=1)

        if isinstance(core_edge, geo.Curve):
            # 플리커링 방지를 위해 0.1 빼줌
            height_to_extrude = height - 0.1
            if reverse:
                height_to_extrude = -height_to_extrude

            return geo.Extrusion.Create(core_edge, height_to_extrude, True)
        return None

    def set_core(self, offset_distance, reverse=False):
        if self.floor is None:
            return
        self.core = self._get_core(offset_distance, reverse)

    def set_slab(self, thickness):
        if self.floor is None:
            return
        ceiling_edges = self.ceiling.DuplicateEdgeCurves()
        ceiling_edge = ghcomp.JoinCurves(ceiling_edges, True)
        self.slab = geo.Extrusion.Create(ceiling_edge, -thickness, True)


class RoofFloor(Floor):
    def __init__(self, brep):
        super().__init__(brep)
        random_height = random.uniform(1.2, 3.6)
        self.top_cores = self._get_core(random.uniform(6, 11), height=random_height)
        self.railing = self._get_railing()

    def _get_railing(self):
        if self.ceiling is None:
            return
        edges = self.ceiling.DuplicateEdgeCurves()
        edge = ghcomp.JoinCurves(edges, True)
        inward_edge = ghcomp.OffsetCurve(edge, -random.uniform(0.2, 0.5), corners=1)
        if isinstance(inward_edge, list):
            inward_edge = inward_edge[0]
        # loft_option = ghcomp.LoftOptions(True, True, 0, 0, 3)
        # loft = ghcomp.Loft([edge, inward_edge], loft_option)

        loft = ghcomp.Loft([edge, inward_edge])
        extrusion = ghcomp.Extrude(loft, -geo.Vector3d.ZAxis * random.uniform(0.5, 1.5))
        extrusion = ghcomp.Extrude(loft, geo.Vector3d.ZAxis * random.uniform(0.5, 1.5))
        return extrusion


class BottomFloor(Floor):
    def __init__(self, brep):
        super().__init__(brep)
        self._additional_bottom_setup()

    def _additional_bottom_setup(self):
        # Add any additional setup for BottomFloor here
        print("Additional setup for BottomFloor")


def trim_crv_from_length(crv, length, reverse=False):
    # type: (geo.Curve, float, bool) -> geo.Curve
    """crv의 시작점부터 lenth까지의 커브를 구한다."""
    is_len_possible, param = crv.LengthParameter(length)
    if not is_len_possible:
        return crv

    ## for Rhino 8, REMOVE FOR RHINO7 !!!##
    param = float(param)
    ########################################

    if reverse:
        return crv.Trim(param, crv.Domain.Max)

    return crv.Trim(0.0, param)


def cluster_breps(breps, threshold=0.5):

    def are_breps_touching(brep1, brep2, threshold):
        for vertex in brep1.Vertices:
            point = vertex.Location
            closest_point = brep2.ClosestPoint(point)
            distance = point.DistanceTo(closest_point)
            if distance < threshold:
                return True
        return False

    def get_all_linked_breps(brep, breps, threshold):
        linked_breps = [brep]
        queue = [brep]
        while queue:
            current_brep = queue.pop(0)
            for other_brep in breps:
                if other_brep not in linked_breps and are_breps_touching(
                    current_brep, other_brep, threshold
                ):
                    linked_breps.append(other_brep)
                    queue.append(other_brep)
        return linked_breps

    clusters = []
    for brep in breps:
        if any(brep in cluster for cluster in clusters):
            continue

        breps_not_in_clusters = [
            b for b in breps if not any(b in cluster for cluster in clusters)
        ]
        cluster = get_all_linked_breps(brep, breps_not_in_clusters, threshold)

        clusters.append(cluster)
    return clusters


def bake_to_layer(objects, layer_name):
    if not objects:
        return
    if not isinstance(objects, list):
        objects = [objects]
    ghdoc = sc.doc
    sc.doc = Rhino.RhinoDoc.ActiveDoc
    try:
        if not rs.IsLayer(layer_name):
            rs.AddLayer(layer_name)
        for obj in objects:
            if not obj:
                continue
            if not isinstance(obj, geo.Brep):
                obj = obj.ToBrep()
            obj_id = sc.doc.Objects.AddBrep(obj)
            obj_ref = sc.doc.Objects.Find(obj_id)
            # obj_id로 FioreObject를 찾을 수 없을 때가 있음
            if not obj_ref:
                continue
            attributes = obj_ref.Attributes
            attributes.LayerIndex = sc.doc.Layers.FindByFullPath(layer_name, True)
            sc.doc.Objects.ModifyAttributes(obj_id, attributes, True)
        sc.doc.Views.Redraw()
    finally:
        sc.doc = ghdoc


def get_top_brep(cluster):
    if not cluster:
        return None
    return max(cluster, key=lambda brep: brep.GetBoundingBox(True).Max.Z)


def get_bottom_brep(cluster):
    if not cluster:
        return None
    return min(cluster, key=lambda brep: brep.GetBoundingBox(True).Min.Z)


print("TEST 1", len(breps))

### input = breps, glass_dist, frame_dist, core_dist, slab_thick
clusters = cluster_breps(breps)

###### COMMON OUTPUT ######
glasses = []
glass_frames = []
cores = []
slabs = []


###### TOP FLOOR OUTPUT ######
railings = []
top_cores = []

###### BOTTOM FLOOR OUTPUT ######
bottom_walls = []
bottoms = []

floors = []
ceilings = []

ornaments = []
for cluster in clusters:
    print("====================")
    print("    cluster : ", len(cluster))
    _glass_dist = glass_dist * random.uniform(0.8, 2.4)
    _frame_dist = frame_dist * random.uniform(0.8, 1.8)
    _core_dist = core_dist * random.uniform(0.8, 1.2)
    _slab_thick = slab_thick * random.uniform(0.8, 1.2)

    # Add to bottom floor output
    bottom_brep = get_bottom_brep(cluster)
    if bottom_brep:
        print("bottom_brep Succed", bottom_brep)
        cluster.remove(bottom_brep)
        bottom_floor = BottomFloor(bottom_brep)
        bottom_walls.extend(bottom_floor.glass)
        bottoms.append(bottom_floor)

    # Add to top floor output
    top_brep = get_top_brep(cluster)
    if top_brep:
        print("top_brep Succed", top_brep)
        # TOP FLOOR 은 옥상을 추가적으로 생성하는 방식이라  층 생성을 제거하지 않는다.
        # cluster.remove(top_brep)
        roof_floor = RoofFloor(top_brep)
        railings.append(roof_floor.railing)
        top_cores.append(roof_floor.top_cores)

    bldg_glasses = []
    bldg_glass_frames = []
    bldg_cores = []
    bldg_slabs = []
    bldg_floors = []
    bldg_ceilings = []
    bldg_ornaments = []
    for brep in cluster:

        try:
            floor = Floor(brep)
            floor.set_glass_frames(_glass_dist, _frame_dist)
            floor.set_core(_core_dist, True)
            floor.set_slab(_slab_thick)

            # Add to common output
            bldg_glasses.extend(floor.glass)
            bldg_glass_frames.extend(floor.glass_frames)
            bldg_cores.append(floor.core)
            bldg_slabs.append(floor.slab)

            bldg_floors.append(floor.floor)
            bldg_ceilings.append(floor.ceiling)

        except Exception as e:
            print(f"Error processing brep: {e}")
            continue

    # 1 : No Ornament, 2 : Vertical Ornament, 3 : Horizontal Ornament, 4 : Both
    random_value = random.choice([1, 2, 3, 4])
    if random_value == 1:
        pass
    # Vertical Ornament : 창문 프레임으로 부터 생성
    elif random_value == 2:
        for brep in bldg_glass_frames:
            ornament = ghcomp.OffsetSurface(brep, random.uniform(0.1, 0.3))
            if ornament:
                bldg_ornaments.append(ornament)
    # Horizontal Ornament : slab옆면으로 부터 생성
    elif random_value == 3:
        for brep in bldg_slabs:
            ornament = ghcomp.OffsetSurface(brep, random.uniform(0.1, 0.3))
            if ornament:
                bldg_ornaments.append(ornament)
    elif random_value == 4:
        for brep in bldg_glass_frames:
            ornament = ghcomp.OffsetSurface(brep, random.uniform(0.1, 0.3))
            if ornament:
                bldg_ornaments.append(ornament)
        for brep in bldg_slabs:
            ornament = ghcomp.OffsetSurface(brep, random.uniform(0.1, 0.3))
            if ornament:
                bldg_ornaments.append(ornament)

    glasses.extend(bldg_glasses)
    glass_frames.extend(bldg_glass_frames)
    cores.extend(bldg_cores)
    slabs.extend(bldg_slabs)
    floors.extend(bldg_floors)
    ceilings.extend(bldg_ceilings)
    ornaments.extend(bldg_ornaments)


### TEST CODE ###
# sc.sticky["floors"] = floors
# sc.sticky["ceilings"] = ceilings

# sc.sticky["glasses"] = glasses
# sc.sticky["glass_frames"] = glass_frames
# sc.sticky["cores"] = cores
# sc.sticky["slabs"] = slabs
# sc.sticky["railings"] = railings
# sc.sticky["top_cores"] = top_cores
# sc.sticky["bottom_walls"] = bottom_walls
# sc.sticky["bottoms"] = bottoms

#################

if bake:
    # Bake to Rhino
    # Common output
    bake_to_layer(glasses, "bake::glasses")
    bake_to_layer(glass_frames, "bake::glass_frames")
    bake_to_layer(cores, "bake::cores")
    bake_to_layer(slabs, "bake::slabs")
    # Top floor output
    bake_to_layer(railings, "bake::railings")
    bake_to_layer(top_cores, "bake::top_cores")
    # Bottom floor output
    bake_to_layer(bottom_walls, "bake::bottom_walls")
    # ornaments output
    bake_to_layer(ornaments, "bake::ornaments")
