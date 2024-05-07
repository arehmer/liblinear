#ifndef _NEWTON_H
#define _NEWTON_H

namespace liblinear {



    class function
    {
    public:
	    virtual float fun(float*w) = 0 ;
	    virtual void grad(float*w, float*g) = 0 ;
	    virtual void Hv(float*s, float*Hs) = 0 ;
	    virtual int get_nr_variable(void) = 0 ;
	    virtual void get_diag_preconditioner(float*M) = 0 ;
	    virtual ~function(void){}

	    // base implementation in newton.cpp
	    virtual float linesearch_and_update(float*w, float *s, float *f, float *g, float alpha);
    };

    class NEWTON
    {
    public:
	    NEWTON(const function *fun_obj, float eps = 0.1, float eps_cg = 0.5, int max_iter = 1000);
	    ~NEWTON();

	    void newton(float *w);
	    void set_print_string(void (*i_print) (const char *buf));

    private:
	    int pcg(float *g, float *M, float *s, float *r);

	    float eps;
	    float eps_cg;
	    int max_iter;
	    function *fun_obj;
	    void info(const char *fmt,...);
	    void (*newton_print_string)(const char *buf);
    };


extern float dnrm2_(int*, float*, int*);
extern float ddot_(int*, float*, int*, float*, int*);
extern int daxpy_(int*, float*, float*, int*, float*, int*);
extern int dscal_(int*, float*, float*, int*);

}//liblinear
#endif
