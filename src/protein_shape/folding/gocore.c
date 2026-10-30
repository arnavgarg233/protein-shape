/* Single-thread Langevin integrator (BAOAB, the OpenMM LangevinMiddle splitting) for the C-alpha Go model of
   gomodel.py (Clementi, Nymeyer & Onuchic 2000, Eq. A1). Units: kJ/mol (eps = 1), nm, ps, amu.
   Compile: cc -O3 -shared -fPIC -o libgocore.so gocore.c -lm (goc.py does this on first use). */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n;
    double mass;
    double *r0, *kb;                 /* bonds i,i+1 : n-1 */
    double *th0; double kth;          /* angles i,i+1,i+2 : n-2 */
    double *phi0; double k1, k3;      /* dihedrals i..i+3 : n-3 */
    int nc; int *ci, *cj; double *cs; /* native contacts */
    char *excl;                       /* n*n : 1 if no repulsion */
    double sig_rep;                   /* 0.4 nm */
    double rcut;                      /* repulsion cutoff */
    int nl; int *li, *lj, nlcap;      /* Verlet list */
    double *xref; double skin;
} Model;

static uint64_t rs[2];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t nextr(void) {  /* xoroshiro128+ */
    uint64_t s0 = rs[0], s1 = rs[1], r = s0 + s1;
    s1 ^= s0; rs[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); rs[1] = rotl(s1, 37); return r;
}
static inline double unif(void) { return ((nextr() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static inline double gauss(void) { return sqrt(-2.0 * log(unif())) * cos(6.283185307179586 * unif()); }
void seed_rng(uint64_t s) {
    uint64_t z = s + 0x9E3779B97F4A7C15ULL;
    for (int k = 0; k < 2; k++) { z += 0x9E3779B97F4A7C15ULL; uint64_t y = z; y = (y ^ (y >> 30)) * 0xBF58476D1CE4E5B9ULL;
        y = (y ^ (y >> 27)) * 0x94D049BB133111EBULL; rs[k] = y ^ (y >> 31); }
    for (int k = 0; k < 20; k++) nextr();
}

Model *model_new(int n, double mass, const double *r0, double kb_openmm, const double *th0, double kth_openmm,
                 const double *phi0, double k1, double k3, int nc, const int *ci, const int *cj, const double *cs,
                 const char *excl, double sig_rep) {
    Model *m = calloc(1, sizeof(Model));
    m->n = n; m->mass = mass; m->kth = kth_openmm; m->k1 = k1; m->k3 = k3; m->nc = nc; m->sig_rep = sig_rep;
    m->r0 = malloc(sizeof(double) * (n - 1)); memcpy(m->r0, r0, sizeof(double) * (n - 1));
    m->kb = malloc(sizeof(double)); m->kb[0] = kb_openmm;
    m->th0 = malloc(sizeof(double) * (n - 2)); memcpy(m->th0, th0, sizeof(double) * (n - 2));
    m->phi0 = malloc(sizeof(double) * (n - 3)); memcpy(m->phi0, phi0, sizeof(double) * (n - 3));
    m->ci = malloc(sizeof(int) * nc); memcpy(m->ci, ci, sizeof(int) * nc);
    m->cj = malloc(sizeof(int) * nc); memcpy(m->cj, cj, sizeof(int) * nc);
    m->cs = malloc(sizeof(double) * nc); memcpy(m->cs, cs, sizeof(double) * nc);
    m->excl = malloc(n * n); memcpy(m->excl, excl, n * n);
    m->rcut = 0.9; m->skin = 0.3;
    m->nlcap = n * n / 2 + 1;
    m->li = malloc(sizeof(int) * m->nlcap); m->lj = malloc(sizeof(int) * m->nlcap);
    m->xref = malloc(sizeof(double) * 3 * n);
    m->nl = -1;
    return m;
}

static void build_list(Model *m, const double *x) {
    int n = m->n, c = 0; double rl = m->rcut + m->skin; rl *= rl;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        if (m->excl[i * n + j]) continue;
        double dx = x[3*i]-x[3*j], dy = x[3*i+1]-x[3*j+1], dz = x[3*i+2]-x[3*j+2];
        if (dx*dx + dy*dy + dz*dz < rl) { m->li[c] = i; m->lj[c] = j; c++; }
    }
    m->nl = c; memcpy(m->xref, x, sizeof(double) * 3 * n);
}

static void check_list(Model *m, const double *x) {
    if (m->nl < 0) { build_list(m, x); return; }
    double lim = 0.25 * m->skin * m->skin;
    for (int i = 0; i < m->n; i++) {
        double dx = x[3*i]-m->xref[3*i], dy = x[3*i+1]-m->xref[3*i+1], dz = x[3*i+2]-m->xref[3*i+2];
        if (dx*dx + dy*dy + dz*dz > lim) { build_list(m, x); return; }
    }
}

/* energies out: e[0]=bonded (bond+angle+dihedral), e[1]=contacts, e[2]=repulsion */
void forces(Model *m, const double *x, double *f, double *e) {
    int n = m->n; double eb = 0, ec = 0, er = 0;
    memset(f, 0, sizeof(double) * 3 * n);
    for (int i = 0; i < n - 1; i++) {
        double d[3] = {x[3*i]-x[3*i+3], x[3*i+1]-x[3*i+4], x[3*i+2]-x[3*i+5]};
        double r = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]), dr = r - m->r0[i];
        eb += 0.5 * m->kb[0] * dr * dr; double g = -m->kb[0] * dr / r;
        for (int a = 0; a < 3; a++) { f[3*i+a] += g * d[a]; f[3*i+3+a] -= g * d[a]; }
    }
    for (int i = 0; i < n - 2; i++) {
        double a[3], b[3];
        for (int k = 0; k < 3; k++) { a[k] = x[3*i+k]-x[3*i+3+k]; b[k] = x[3*i+6+k]-x[3*i+3+k]; }
        double la = sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]), lb = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
        double c = (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]) / (la * lb); if (c > 1) c = 1; if (c < -1) c = -1;
        double th = acos(c), dth = th - m->th0[i]; eb += 0.5 * m->kth * dth * dth;
        double s = sqrt(1 - c*c); if (s < 1e-8) s = 1e-8;
        double dE = m->kth * dth;           /* dE/dtheta ; dtheta/dx = -(1/s) dc/dx */
        for (int k = 0; k < 3; k++) {
            double dca = (b[k]/(la*lb) - c * a[k]/(la*la));
            double dcb = (a[k]/(la*lb) - c * b[k]/(lb*lb));
            double fa = dE / s * dca, fb = dE / s * dcb;   /* F = -dE/dx = +dE/s * dc/dx */
            f[3*i+k] += fa; f[3*i+6+k] += fb; f[3*i+3+k] -= fa + fb;
        }
    }
    for (int i = 0; i < n - 3; i++) {
        const double *xi = x+3*i, *xj = x+3*i+3, *xk = x+3*i+6, *xl = x+3*i+9;
        double rij[3], rkj[3], rkl[3], mv[3], nv[3];
        for (int k = 0; k < 3; k++) { rij[k] = xi[k]-xj[k]; rkj[k] = xk[k]-xj[k]; rkl[k] = xk[k]-xl[k]; }
        mv[0] = rij[1]*rkj[2]-rij[2]*rkj[1]; mv[1] = rij[2]*rkj[0]-rij[0]*rkj[2]; mv[2] = rij[0]*rkj[1]-rij[1]*rkj[0];
        nv[0] = rkj[1]*rkl[2]-rkj[2]*rkl[1]; nv[1] = rkj[2]*rkl[0]-rkj[0]*rkl[2]; nv[2] = rkj[0]*rkl[1]-rkj[1]*rkl[0];
        double m2 = mv[0]*mv[0]+mv[1]*mv[1]+mv[2]*mv[2], n2 = nv[0]*nv[0]+nv[1]*nv[1]+nv[2]*nv[2];
        double nrkj2 = rkj[0]*rkj[0]+rkj[1]*rkj[1]+rkj[2]*rkj[2], nrkj = sqrt(nrkj2);
        double cphi = mv[0]*nv[0]+mv[1]*nv[1]+mv[2]*nv[2];
        double sphi = nrkj * (rij[0]*nv[0]+rij[1]*nv[1]+rij[2]*nv[2]);
        double phi = atan2(sphi, cphi), d = phi - m->phi0[i];
        eb += m->k1 * (1 - cos(d)) + m->k3 * (1 - cos(3*d));
        double ddphi = m->k1 * sin(d) + 3 * m->k3 * sin(3*d);
        double p = (rij[0]*rkj[0]+rij[1]*rkj[1]+rij[2]*rkj[2]) / nrkj2;
        double q = (rkl[0]*rkj[0]+rkl[1]*rkj[1]+rkl[2]*rkj[2]) / nrkj2;
        for (int k = 0; k < 3; k++) {
            double fi = -ddphi * nrkj / m2 * mv[k], fl = ddphi * nrkj / n2 * nv[k];
            double sv = p * fi - q * fl, fj = fi - sv, fk = fl + sv;
            f[3*i+k] += fi; f[3*i+3+k] -= fj; f[3*i+6+k] -= fk; f[3*i+9+k] += fl;
        }
    }
    for (int c = 0; c < m->nc; c++) {
        int i = m->ci[c], j = m->cj[c];
        double d[3] = {x[3*i]-x[3*j], x[3*i+1]-x[3*j+1], x[3*i+2]-x[3*j+2]};
        double r2 = d[0]*d[0]+d[1]*d[1]+d[2]*d[2], r = sqrt(r2), q = m->cs[c] / r;
        double q2 = q*q, q4 = q2*q2, q10 = q4*q4*q2, q12 = q10*q2;
        ec += 5*q12 - 6*q10; double dEdr = (-60*q12 + 60*q10) / r, g = -dEdr / r;
        for (int a = 0; a < 3; a++) { f[3*i+a] += g * d[a]; f[3*j+a] -= g * d[a]; }
    }
    check_list(m, x);
    double rc2 = m->rcut * m->rcut, s12 = pow(m->sig_rep, 12);
    for (int l = 0; l < m->nl; l++) {
        int i = m->li[l], j = m->lj[l];
        double d[3] = {x[3*i]-x[3*j], x[3*i+1]-x[3*j+1], x[3*i+2]-x[3*j+2]};
        double r2 = d[0]*d[0]+d[1]*d[1]+d[2]*d[2]; if (r2 > rc2) continue;
        double ir2 = 1.0 / r2, ir6 = ir2*ir2*ir2, e12 = s12 * ir6 * ir6;
        er += e12; double g = 12 * e12 * ir2;
        for (int a = 0; a < 3; a++) { f[3*i+a] += g * d[a]; f[3*j+a] -= g * d[a]; }
    }
    e[0] = eb; e[1] = ec; e[2] = er;
}

/* BAOAB / "middle" Langevin; kT in kJ/mol, gamma 1/ps, dt ps. x,v updated in place. */
void run(Model *m, double *x, double *v, int nsteps, double kT, double gamma, double dt, double *ebuf) {
    int n = m->n; double *f = malloc(sizeof(double) * 3 * n); double mass = m->mass;
    double c1 = exp(-gamma * dt), c2 = sqrt((1 - c1*c1) * kT / mass);
    forces(m, x, f, ebuf);
    for (int s = 0; s < nsteps; s++) {
        for (int k = 0; k < 3*n; k++) { v[k] += 0.5 * dt * f[k] / mass; x[k] += 0.5 * dt * v[k]; }
        for (int k = 0; k < 3*n; k++) v[k] = c1 * v[k] + c2 * gauss();
        for (int k = 0; k < 3*n; k++) x[k] += 0.5 * dt * v[k];
        forces(m, x, f, ebuf);
        for (int k = 0; k < 3*n; k++) v[k] += 0.5 * dt * f[k] / mass;
    }
    free(f);
}
