#include <basic_types.hpp>
#include <cmath>
#include <kokkos_abstraction.hpp>

#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Profiling.hpp"
#include "old_abstractions.hpp"

using parthenon::Real;
using Arr_t = Kokkos::View<Real *****>;

struct par_outer_old {
  template <typename... Args> static void par_for_outer(Args &&...args) {
    parthenon::old::par_for_outer(std::forward<Args>(args)...);
  }
};

struct par_inner_old {
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION static void par_for_inner(Args &&...args) {
    parthenon::old::par_for_inner(std::forward<Args>(args)...);
  }
};

struct par_outer_new {
  template <typename... Args> static void par_for_outer(Args &&...args) {
    parthenon::par_for_outer(std::forward<Args>(args)...);
  }
};

struct par_inner_new {
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION static void par_for_inner(Args &&...args) {
    parthenon::par_for_inner(std::forward<Args>(args)...);
  }
};

template <typename OUTER, typename INNER>
void KernelTest(OUTER par_for_outer, INNER par_for_inner, Arr_t U, Arr_t F) {
  const int nb = U.extent(0);
  const int nvar = U.extent(1);
  const int nx3 = U.extent(2);
  const int nx2 = U.extent(3);
  const int nx1 = U.extent(4);

  const int scratch_level = 0;
  std::size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(nvar, nx1);

  const Real dx = 0.1;
  OUTER::par_for_outer(
      PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes, scratch_level, 0, nb - 1,
      0, nx3 - 1, 0, nx2 - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k,
                    const int j) {
        parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level),
                                         nvar, nx1);
        parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level),
                                         nvar, nx1);

        const Real z = 0. + dx * k;
        const Real y = 0. + dx * j;
        INNER::par_for_inner(
            member, 0, nvar - 1, 0, nx1 - 1, [&](const int var, const int i) {
              const Real x = 0. + dx * i;
              ql(var, i) =
                  b * std::cos(x - 0.5 * dx) * std::cos(y) * std::cos(z);
              qr(var, i) =
                  b * std::cos(x + 0.5 * dx) * std::cos(y) * std::cos(z);
            });

        member.team_barrier();

        INNER::par_for_inner(
            member, 0, nvar - 1, 0, nx1 - 1, [&](const int var, const int i) {
              F(b, var, k, j, i) = ql(var, i) * std::exp(qr(var, i));
            });
      });
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    Arr_t U, F;

    const int nb = 10;
    const int nx3 = 32;
    const int nx2 = 64;
    const int nx1 = 64;
    const int nvar = 5;

    U = Arr_t("U", nb, nvar, nx3, nx2, nx1);
    F = Arr_t("F", nb, nvar, nx3, nx2, nx1);

    const int niter = 100;
    for (int i = 0; i < niter; i++) {

      Kokkos::Profiling::pushRegion("newnew");
      KernelTest(par_outer_new(), par_inner_new(), U, F);
      Kokkos::Profiling::popRegion();

      Kokkos::Profiling::pushRegion("oldnew");
      KernelTest(par_outer_old(), par_inner_new(), U, F);
      Kokkos::Profiling::popRegion();

      Kokkos::Profiling::pushRegion("oldold");
      KernelTest(par_outer_old(), par_inner_old(), U, F);
      Kokkos::Profiling::popRegion();

      Kokkos::Profiling::pushRegion("newold");
      KernelTest(par_outer_new(), par_inner_old(), U, F);
      Kokkos::Profiling::popRegion();
    }
  }

  Kokkos::finalize();
  return (0);
}
