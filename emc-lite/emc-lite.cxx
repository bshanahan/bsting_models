/*
 * 2D temperature conduction problem
 *
 */

#include <bout/fv_ops.hxx>
#include <bout/physicsmodel.hxx>

class emclite : public PhysicsModel {
private:
  Field3D T, n; // Evolving temperature equation only

  BoutReal chi; // Parallel conduction coefficient
  BoutReal Te, kappa_epar, kappa_e0;
  Field3D chin; // cached chi * n
  Field3D sheath_dT;
  BoutReal Nt, Cs0, Tt, sheath_gamma;
  BoutReal q;
  bool parallel_sheaths;
  Coordinates *coord;

protected:

  void check_all(Field3D &f) {
    checkData(f);
    checkData(f.yup());
    checkData(f.ydown());
  }

  Field3D mul_all(const Field3D &a, const Field3D &b) {
    Field3D result = a * b;
    result.splitParallelSlices();
    result.yup() = a.yup() * b.yup();
    result.ydown() = a.ydown() * b.ydown();
    return result;
  }

  // This is called once at the start
  int init(bool UNUSED(restarting)) override {

    // Get the options
    auto &options = Options::root()["emc-lite"];

    // Read from BOUT.inp, setting default to 1.0
    // The doc() provides some documentation in BOUT.settings
    chi = options["chi"].doc("Perpendicular coefficient").withDefault(1.0);
    Te = options["Te"].doc("Electron Temperature").withDefault(20.0);
    n = options["n"].doc("Density").withDefault(1.0);
    kappa_e0 = options["kappa_e0"].doc("parallel conduction coefficient").withDefault(1.0);
    kappa_epar = kappa_e0 * pow(Te, 5 / 2);

    parallel_sheaths = options["parallel_sheaths"]
                           .doc("parallel boundary conditions")
                           .withDefault(false);
    Nt = options["Nt"].doc("Density at target").withDefault(1.0);
    Tt = options["Tt"].doc("Temperature at target").withDefault(20.0);
    Cs0 = options["Cs"].doc("Sound speed").withDefault(1.0);
    sheath_gamma = options["sheath_gamma"].doc("Sheath Gamma").withDefault(5.5);

    // Heat flux
    q = sheath_gamma * Tt * Nt * Cs0;

    // Tell BOUT++ to solve T
    SOLVE_FOR(T);
    // chi.applyParallelBoundary("parallel_neumann");
    // kappa_epar.applyParallelBoundary("parallel_neumann");
    chin = chi * n;          // mul_all(chi,n);
    mesh->communicate(chin); // Communicate guard cells
    chin.applyParallelBoundary("parallel_neumann");

    coord = mesh->getCoordinates();
    mesh->communicate(coord->g23, coord->g_23, coord->dy, coord->dz, coord->Bxy,
                      coord->J);

    coord->dz.applyParallelBoundary("parallel_neumann");
    coord->dy.applyParallelBoundary("parallel_neumann");
    coord->J.applyParallelBoundary("parallel_neumann");
    // coord->g_22.applyParallelBoundary("parallel_neumann");
    coord->g_23.applyParallelBoundary("parallel_neumann");
    coord->g23.applyParallelBoundary("parallel_neumann");
    coord->Bxy.applyParallelBoundary("parallel_neumann");

    SAVE_ONCE(chi, n, Te, kappa_epar);
    return 0;
  }

  int rhs(BoutReal time) override {
    printf("TIME = %e\r", time);

    mesh->communicate(T);
    T.applyParallelBoundary("parallel_neumann");
    ddt(T) = Div_par_K_Grad_par(
        kappa_epar, T); // Parallel diffusion Div_{||}( chi * Grad_{||}(T) )
    ddt(T) += FV::Div_a_Laplace_perp(chin, T);

    if (parallel_sheaths) {
      for (const auto &bndry_par :
           mesh->getBoundariesPar(BoundaryParType::xout)) {
        // Sound speed (normalised units)
        BoutReal Cs = Cs0 * bndry_par->dir; //* sqrt(tesheath + tisheath);

        for (bndry_par->first(); !bndry_par->isDone(); bndry_par->next()) {
	  int x = bndry_par->x; int y = bndry_par->y; int z = bndry_par->z;

          // Temperature and density at the sheath entrance	  
	  // Multiply by cell area to get power
	  BoutReal flux =
	    q
	    * (coord->J(x, y, z) + coord->J.ynext(bndry_par->dir)(x, y + bndry_par->dir, z))
	    / (sqrt(coord->g_22(x, y, z))
	       + sqrt(coord->g_22.ynext(bndry_par->dir)(x, y + bndry_par->dir, z)));
	  	  // Divide by volume of cell, and 2/3 to get pressure
	  BoutReal power =
	    flux
	    / (coord->dy(x, y, z) * coord->J(x, y, z));
	  sheath_dT(x, y, z) -= power;
	}
      }
      ddt(T) += sheath_dT;
    }
     
    return 0;
  }
  
};

BOUTMAIN(emclite);
