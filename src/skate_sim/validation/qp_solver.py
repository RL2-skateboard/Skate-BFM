"""Independent numerical backend for the unchanged experimental convex QP."""
import numpy as np
from scipy import sparse
from types import SimpleNamespace


def clarabel_solve(P,q,A,lower,upper,tolerance=1e-9,diagnostic_only=False,static_regularization=None):
    try:import clarabel
    except ImportError as error:raise RuntimeError('Experimental QP requires explicit clarabel==0.11.1 installation; no approximate-action fallback') from error
    finite_lower=np.isfinite(lower);finite_upper=np.isfinite(upper)
    equal=finite_lower&finite_upper&(abs(upper-lower)<1e-12)
    iu=np.flatnonzero(finite_upper&~equal);il=np.flatnonzero(finite_lower&~equal);ie=np.flatnonzero(equal)
    matrix=np.vstack([A[ie],A[iu],-A[il]]);rhs=np.r_[upper[ie],upper[iu],-lower[il]]
    cones=[]
    if len(ie):cones.append(clarabel.ZeroConeT(len(ie)))
    if len(iu)+len(il):cones.append(clarabel.NonnegativeConeT(len(iu)+len(il)))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200
    if static_regularization is not None:settings.static_regularization_constant=static_regularization
    settings.tol_gap_abs=tolerance;settings.tol_gap_rel=tolerance;settings.tol_feas=tolerance
    solution=clarabel.DefaultSolver(sparse.csc_matrix(np.triu(.5*(P+P.T))),q,sparse.csc_matrix(matrix),rhs,cones,settings).solve()
    if solution.status!=clarabel.SolverStatus.Solved and not diagnostic_only:raise RuntimeError(f'Independent Clarabel QP {solution.status}; primal={solution.r_prim}, dual={solution.r_dual}')
    dual=np.zeros(len(lower));z=np.asarray(solution.z);dual[ie]=z[:len(ie)];dual[iu]+=z[len(ie):len(ie)+len(iu)];dual[il]-=z[len(ie)+len(iu):]
    x=np.asarray(solution.x);gradient=P@x+q+A.T@dual
    return SimpleNamespace(x=x,y=dual,info=SimpleNamespace(status='solved' if solution.status==clarabel.SolverStatus.Solved else str(solution.status),iter=solution.iterations,pri_res=solution.r_prim,dua_res=solution.r_dual),backend='clarabel',version=clarabel.__version__,module_file=clarabel.__file__,gradient_inf=float(np.max(abs(gradient))))


def clarabel_least_squares(K,target,A,lower,upper):
    """Exact residual lifting; avoids forming the ill-conditioned normal matrix."""
    import clarabel
    n=K.shape[1];m=K.shape[0]
    iu=np.flatnonzero(np.isfinite(upper));il=np.flatnonzero(np.isfinite(lower))
    equation=sparse.hstack([-sparse.csc_matrix(K),sparse.eye(m,format='csc')],format='csc')
    inequalities=sparse.vstack([sparse.hstack([sparse.csc_matrix(A[iu]),sparse.csc_matrix((len(iu),m))]),sparse.hstack([-sparse.csc_matrix(A[il]),sparse.csc_matrix((len(il),m))])],format='csc')
    matrix=sparse.vstack([equation,inequalities],format='csc');rhs=np.r_[-target,upper[iu],-lower[il]]
    P=sparse.diags(np.r_[np.zeros(n),np.full(m,2.)],format='csc');q=np.zeros(n+m)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200
    settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9;settings.tol_feas=1e-9
    settings.static_regularization_constant=1e-10
    solution=clarabel.DefaultSolver(P,q,matrix,rhs,[clarabel.ZeroConeT(m),clarabel.NonnegativeConeT(len(iu)+len(il))],settings).solve()
    first_status=str(solution.status);internal_gap_tolerance=1e-9;numeric_retries=0
    if solution.status==clarabel.SolverStatus.AlmostSolved:
        settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8
        solution=clarabel.DefaultSolver(P,q,matrix,rhs,[clarabel.ZeroConeT(m),clarabel.NonnegativeConeT(len(iu)+len(il))],settings).solve()
        internal_gap_tolerance=1e-8;numeric_retries=1
    if solution.status!=clarabel.SolverStatus.Solved:raise RuntimeError(f'Lifted Clarabel LS {solution.status}; primal={solution.r_prim}, dual={solution.r_dual}')
    z=np.asarray(solution.z);dual=np.zeros(len(lower));dual[iu]+=z[m:m+len(iu)];dual[il]-=z[m+len(iu):]
    return SimpleNamespace(x=np.asarray(solution.x[:n]),y=dual,info=SimpleNamespace(status='solved',iter=solution.iterations,pri_res=solution.r_prim,dua_res=solution.r_dual),backend='clarabel_lifted_ls',first_status=first_status,internal_gap_tolerance=internal_gap_tolerance,numeric_retries=numeric_retries,version=clarabel.__version__,module_file=clarabel.__file__,primal_objective=solution.obj_val,dual_objective=solution.obj_val_dual,relative_dual_gap=abs(solution.obj_val-solution.obj_val_dual)/max(1.,abs(solution.obj_val)),dual_complementarity_inf=float(np.max(abs(np.asarray(solution.z)*np.asarray(solution.s)))),residual_equation_inf=float(np.max(abs(K@np.asarray(solution.x[:n])-target-np.asarray(solution.x[n:])))))
