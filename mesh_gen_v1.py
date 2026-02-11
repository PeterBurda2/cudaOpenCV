import gmsh
import sys


step_file = "test_cube.step"      # your STEP file
out_msh   = "mesh.msh"           # output .msh (v4.1 is best for downstream tools)
mesh_size = 0.005                  # 0.0 means "use CAD sizing"; else set e.g. 0.02
order     = 1                    # 1: linear, 2: quadratic, 3: cubic

gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setString('Geometry.OCCTargetUnit', 'M')
gmsh.model.add("cad_from_step")

# Import STEP into OpenCASCADE kernel
gmsh.model.occ.importShapes(step_file)
gmsh.model.occ.synchronize()



surfaces = gmsh.model.occ.getEntities(dim=2)
xmin, ymin, zmin =  1e30,  1e30,  1e30
xmax, ymax, zmax = -1e30, -1e30, -1e30
for _, s in surfaces:
    
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = gmsh.model.getBoundingBox(2, s)
    xmin, ymin, zmin = min(xmin, bxmin), min(ymin, bymin), min(zmin, bzmin)
    xmax, ymax, zmax = max(xmax, bxmax), max(ymax, bymax), max(zmax, bzmax)

tol = 1e-6 * max(1.0, xmax - xmin, ymax - ymin, zmax - zmin)

print(f"tol = {tol}")
dirichlet_surfs = []
robin_surfs = []

for _, s in surfaces:
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = gmsh.model.getBoundingBox(2, s)

    # Face at x = xmax (its bounding box has bxmax ~ xmax and bxmin ~ xmax)
    print(f"{abs(bxmax -xmin) }, {abs(bxmin - xmin)}")


    if abs(bxmax - xmin) < tol and abs(bxmin - xmin) < tol:
        dirichlet_surfs.append(s)
    else:
        robin_surfs.append(s)

DIRICHLET_ID = 11
ROBIN_ID = 12

if len(dirichlet_surfs) != 1:
    raise RuntimeError(f"Expected 1 Dirichlet face, found {len(dirichlet_surfs)}. Check cube alignment/tolerance.")


gmsh.model.addPhysicalGroup(2, dirichlet_surfs, tag=DIRICHLET_ID)
gmsh.model.setPhysicalName(2, DIRICHLET_ID, "Dirichlet")

gmsh.model.addPhysicalGroup(2, robin_surfs, tag=ROBIN_ID)
gmsh.model.setPhysicalName(2, ROBIN_ID, "Robin")


vols = gmsh.model.occ.getEntities(dim=3)
DOMAIN_ID = 1
gmsh.model.addPhysicalGroup(3, [v for _, v in vols], tag=DOMAIN_ID)
gmsh.model.setPhysicalName(3, DOMAIN_ID, "Omega")


# Generate 3D mesh
gmsh.model.mesh.generate(3)
gmsh.model.mesh.setOrder(order)

# Write as MSH 4.1 for best compatibility
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write(out_msh)
gmsh.finalize()

# gmsh.option.setNumber("Geometry.SurfaceNumbers", 1)  # show surface entity IDs
# gmsh.option.setNumber("Mesh.SurfaceFaces", 1)        # draw surface mesh
# gmsh.option.setNumber("Mesh.VolumeFaces", 0)         # optional: hide interior faces

# # Colour by physical groups (works once a mesh exists)
# gmsh.option.setNumber("Mesh.ColorCarousel", 2)       # helps distinguish groups (optional)

# gmsh.fltk.run()

# print(f"{surfaces}, {vols}, {vol_tags}")