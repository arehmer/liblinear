#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "newton.h"

namespace liblinear {

#ifndef min
    template <class T> static inline T min(T x, T y) { return (x < y) ? x : y; }
#endif

#ifndef max
    template <class T> static inline T max(T x, T y) { return (x > y) ? x : y; }
#endif

float dnrm2_(int* n, float* x, int* incx)
{
    long int ix, nn, iincx;
    float norm, scale, absxi, ssq, temp;

    /*  DNRM2 returns the euclidean norm of a vector via the function
        name, so that

            DNRM2 := sqrt( x'*x )

        -- This version written on 25-October-1982.
            Modified on 14-October-1993 to inline the call to SLASSQ.
            Sven Hammarling, Nag Ltd.   */

            /* Dereference inputs */
    nn = *n;
    iincx = *incx;

    if (nn > 0 && iincx > 0)
    {
        if (nn == 1)
        {
            norm = fabs(x[0]);
        }
        else
        {
            scale = 0.0;
            ssq = 1.0;

            /* The following loop is equivalent to this call to the LAPACK
                auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */

            for (ix = (nn - 1) * iincx; ix >= 0; ix -= iincx)
            {
                if (x[ix] != 0.0)
                {
                    absxi = fabs(x[ix]);
                    if (scale < absxi)
                    {
                        temp = scale / absxi;
                        ssq = ssq * (temp * temp) + 1.0;
                        scale = absxi;
                    }
                    else
                    {
                        temp = absxi / scale;
                        ssq += temp * temp;
                    }
                }
            }
            norm = scale * sqrt(ssq);
        }
    }
    else
        norm = 0.0;

    return norm;
} // dnrm2_

float ddot_(int* n, float* sx, int* incx, float* sy, int* incy)
{
    long int i, m, nn, iincx, iincy;
    float stemp;
    long int ix, iy;

    /* forms the dot product of two vectors.
       uses unrolled loops for increments equal to one.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*) */

       /* Dereference inputs */
    nn = *n;
    iincx = *incx;
    iincy = *incy;

    stemp = 0.0;
    if (nn > 0)
    {
        if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
        {
            m = nn - 4;
            for (i = 0; i < m; i += 5)
                stemp += sx[i] * sy[i] + sx[i + 1] * sy[i + 1] + sx[i + 2] * sy[i + 2] +
                sx[i + 3] * sy[i + 3] + sx[i + 4] * sy[i + 4];

            for (; i < nn; i++)        /* clean-up loop */
                stemp += sx[i] * sy[i];
        }
        else /* code for unequal increments or equal increments not equal to 1 */
        {
            ix = 0;
            iy = 0;
            if (iincx < 0)
                ix = (1 - nn) * iincx;
            if (iincy < 0)
                iy = (1 - nn) * iincy;
            for (i = 0; i < nn; i++)
            {
                stemp += sx[ix] * sy[iy];
                ix += iincx;
                iy += iincy;
            }
        }
    }

    return stemp;
} /* ddot_ */

int daxpy_(int* n, float* sa, float* sx, int* incx, float* sy,
    int* incy)
{
    long int i, m, ix, iy, nn, iincx, iincy;
    register float ssa;

    /* constant times a vector plus a vector.
       uses unrolled loop for increments equal to one.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*) */

       /* Dereference inputs */
    nn = *n;
    ssa = *sa;
    iincx = *incx;
    iincy = *incy;

    if (nn > 0 && ssa != 0.0)
    {
        if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
        {
            m = nn - 3;
            for (i = 0; i < m; i += 4)
            {
                sy[i] += ssa * sx[i];
                sy[i + 1] += ssa * sx[i + 1];
                sy[i + 2] += ssa * sx[i + 2];
                sy[i + 3] += ssa * sx[i + 3];
            }
            for (; i < nn; ++i) /* clean-up loop */
                sy[i] += ssa * sx[i];
        }
        else /* code for unequal increments or equal increments not equal to 1 */
        {
            ix = iincx >= 0 ? 0 : (1 - nn) * iincx;
            iy = iincy >= 0 ? 0 : (1 - nn) * iincy;
            for (i = 0; i < nn; i++)
            {
                sy[iy] += ssa * sx[ix];
                ix += iincx;
                iy += iincy;
            }
        }
    }

    return 0;
} /* daxpy_ */

int dscal_(int* n, float* sa, float* sx, int* incx)
{
    long int i, m, nincx, nn, iincx;
    float ssa;

    /* scales a vector by a constant.
       uses unrolled loops for increment equal to 1.
       jack dongarra, linpack, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*) */

       /* Dereference inputs */
    nn = *n;
    iincx = *incx;
    ssa = *sa;

    if (nn > 0 && iincx > 0)
    {
        if (iincx == 1) /* code for increment equal to 1 */
        {
            m = nn - 4;
            for (i = 0; i < m; i += 5)
            {
                sx[i] = ssa * sx[i];
                sx[i + 1] = ssa * sx[i + 1];
                sx[i + 2] = ssa * sx[i + 2];
                sx[i + 3] = ssa * sx[i + 3];
                sx[i + 4] = ssa * sx[i + 4];
            }
            for (; i < nn; ++i) /* clean-up loop */
                sx[i] = ssa * sx[i];
        }
        else /* code for increment not equal to 1 */
        {
            nincx = nn * iincx;
            for (i = 0; i < nincx; i += iincx)
                sx[i] = ssa * sx[i];
        }
    }

    return 0;
} /* dscal_ */


    static void default_print(const char* buf)
    {
        fputs(buf, stdout);
        fflush(stdout);
    }

    // On entry *f must be the function value of w
    // On exit w is updated and *f is the new function value
    float function::linesearch_and_update(float* w, float* s, float* f, float* g, float alpha)
    {
        float gTs = 0;
        float eta = 0.01;
        int n = get_nr_variable();
        int max_num_linesearch = 20;
        float* w_new = new float[n];
        float fold = *f;

        for (int i = 0;i < n;i++)
            gTs += s[i] * g[i];

        int num_linesearch = 0;
        for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++)
        {
            for (int i = 0;i < n;i++)
                w_new[i] = w[i] + alpha * s[i];
            *f = fun(w_new);
            if (*f - fold <= eta * alpha * gTs)
                break;
            else
                alpha *= 0.5;
        }

        if (num_linesearch >= max_num_linesearch)
        {
            *f = fold;
            return 0;
        }
        else
            memcpy(w, w_new, sizeof(float) * n);

        delete[] w_new;
        return alpha;
    }

    void NEWTON::info(const char* fmt, ...)
    {
        char buf[BUFSIZ];
        va_list ap;
        va_start(ap, fmt);
        vsprintf(buf, fmt, ap);
        va_end(ap);
        (*newton_print_string)(buf);
    }

    NEWTON::NEWTON(const function* fun_obj, float eps, float eps_cg, int max_iter)
    {
        this->fun_obj = const_cast<function*>(fun_obj);
        this->eps = eps;
        this->eps_cg = eps_cg;
        this->max_iter = max_iter;
        newton_print_string = default_print;
    }

    NEWTON::~NEWTON()
    {
    }

    void NEWTON::newton(float* w)
    {
        int n = fun_obj->get_nr_variable();
        int i, cg_iter;
        float step_size;
        float f, fold, actred;
        float init_step_size = 1;
        int search = 1, iter = 1, inc = 1;
        float* s = new float[n];
        float* r = new float[n];
        float* g = new float[n];

        const float alpha_pcg = 0.01;
        float* M = new float[n];

        // calculate gradient norm at w=0 for stopping condition.
        float* w0 = new float[n];
        for (i = 0; i < n; i++)
            w0[i] = 0;
        fun_obj->fun(w0);
        fun_obj->grad(w0, g);
        float gnorm0 = dnrm2_(&n, g, &inc);
        delete[] w0;

        f = fun_obj->fun(w);
        fun_obj->grad(w, g);
        float gnorm = dnrm2_(&n, g, &inc);
        info("init f %5.3e |g| %5.3e\n", f, gnorm);

        if (gnorm <= eps * gnorm0)
            search = 0;

        while (iter <= max_iter && search)
        {
            fun_obj->get_diag_preconditioner(M);
            for (i = 0; i < n; i++)
                M[i] = (1 - alpha_pcg) + alpha_pcg * M[i];
            cg_iter = pcg(g, M, s, r);

            fold = f;
            step_size = fun_obj->linesearch_and_update(w, s, &f, g, init_step_size);

            if (step_size == 0)
            {
                info("WARNING: line search fails\n");
                break;
            }

            fun_obj->grad(w, g);
            gnorm = dnrm2_(&n, g, &inc);

            info("iter %2d f %5.3e |g| %5.3e CG %3d step_size %4.2e \n", iter, f, gnorm, cg_iter, step_size);

            if (gnorm <= eps * gnorm0)
                break;
            if (f < -1.0e+32)
            {
                info("WARNING: f < -1.0e+32\n");
                break;
            }
            actred = fold - f;
            if (fabs(actred) <= 1.0e-12 * fabs(f))
            {
                info("WARNING: actred too small\n");
                break;
            }

            iter++;
        }

        if (iter >= max_iter)
            info("\nWARNING: reaching max number of Newton iterations\n");

        delete[] g;
        delete[] r;
        delete[] s;
        delete[] M;
    }

    int NEWTON::pcg(float* g, float* M, float* s, float* r)
    {
        int i, inc = 1;
        int n = fun_obj->get_nr_variable();
        float one = 1;
        float* d = new float[n];
        float* Hd = new float[n];
        float zTr, znewTrnew, alpha, beta, cgtol, dHd;
        float* z = new float[n];
        float Q = 0, newQ, Qdiff;

        for (i = 0; i < n; i++)
        {
            s[i] = 0;
            r[i] = -g[i];
            z[i] = r[i] / M[i];
            d[i] = z[i];
        }

        zTr = ddot_(&n, z, &inc, r, &inc);
        float gMinv_norm = sqrt(zTr);
        cgtol = min(eps_cg, (float)sqrt(gMinv_norm));
        int cg_iter = 0;
        int max_cg_iter = max(n, 5);

        while (cg_iter < max_cg_iter)
        {
            cg_iter++;

            fun_obj->Hv(d, Hd);
            dHd = ddot_(&n, d, &inc, Hd, &inc);
            // avoid 0/0 in getting alpha
            if (dHd <= 1.0e-16)
                break;

            alpha = zTr / dHd;
            daxpy_(&n, &alpha, d, &inc, s, &inc);
            alpha = -alpha;
            daxpy_(&n, &alpha, Hd, &inc, r, &inc);

            // Using quadratic approximation as CG stopping criterion
            newQ = -0.5 * (ddot_(&n, s, &inc, r, &inc) - ddot_(&n, s, &inc, g, &inc));
            Qdiff = newQ - Q;
            if (newQ <= 0 && Qdiff <= 0)
            {
                if (cg_iter * Qdiff >= cgtol * newQ)
                    break;
            }
            else
            {
                info("WARNING: quadratic approximation > 0 or increasing in CG\n");
                break;
            }
            Q = newQ;

            for (i = 0; i < n; i++)
                z[i] = r[i] / M[i];
            znewTrnew = ddot_(&n, z, &inc, r, &inc);
            beta = znewTrnew / zTr;
            dscal_(&n, &beta, d, &inc);
            daxpy_(&n, &one, z, &inc, d, &inc);
            zTr = znewTrnew;
        }

        if (cg_iter == max_cg_iter)
            info("WARNING: reaching maximal number of CG steps\n");

        delete[] d;
        delete[] Hd;
        delete[] z;

        return cg_iter;
    }

    void NEWTON::set_print_string(void (*print_string) (const char* buf))
    {
        newton_print_string = print_string;
    }

} //liblinear
