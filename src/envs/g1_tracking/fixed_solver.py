"""A process-local reverse-mode-safe outer loop for the MJX solver."""

from contextlib import contextmanager

import jax
from mujoco.mjx._src import solver as _solver

CONVERGENCE_SCAN = "convergence-scan-v1"
FIXED_SOLVER_AND_LINESEARCH_SCAN = "fixed-solver-and-linesearch-scan-v1"
_SUPPORTED_SEMANTICS = frozenset((CONVERGENCE_SCAN, FIXED_SOLVER_AND_LINESEARCH_SCAN))
_ACTIVE_SEMANTIC: str | None = None


def _fixed_loop_scan(_cond_fun, body_fun, init_val, max_iter):
    """Execute exactly ``max_iter`` loop bodies with a differentiable scan."""
    return jax.lax.scan(
        lambda current, _: (body_fun(current), None),
        init_val,
        xs=None,
        length=max_iter,
    )[0]


def active_solver_gradient_semantic() -> str | None:
    """Return the process-local semantic active inside the scoped context."""
    return _ACTIVE_SEMANTIC


def _solve_with_fixed_outer_loop(model, data):
    """Match stock MJX solve while expressing its bounded outer loop as scan."""
    if not isinstance(model.opt._impl, _solver.OptionJAX):
        raise TypeError("solve requires JAX backend implementation")

    def cond(ctx):
        improvement = _solver._rescale(model, ctx.prev_cost - ctx.cost)
        gradient = _solver._rescale(model, _solver.math.norm(ctx.grad))
        done = ctx.solver_niter >= model.opt.iterations
        done |= improvement < model.opt.tolerance
        done |= gradient < model.opt.tolerance
        return ~done

    def body(ctx):
        ctx = _solver._linesearch(model, data, ctx)
        prev_grad, prev_mgrad = ctx.grad, ctx.Mgrad
        ctx = _solver._update_constraint(model, data, ctx)
        ctx = _solver._update_gradient(model, data, ctx)

        if model.opt.solver == _solver.SolverType.NEWTON:
            search = -ctx.Mgrad
        else:
            beta = _solver.jp.dot(ctx.grad, ctx.Mgrad - prev_mgrad)
            beta /= _solver.jp.maximum(
                _solver.mujoco.mjMINVAL,
                _solver.jp.dot(prev_grad, prev_mgrad),
            )
            beta = _solver.jp.maximum(0, beta)
            search = -ctx.Mgrad + beta * ctx.search
        return ctx.replace(
            search=search,
            solver_niter=ctx.solver_niter + 1,
        )

    qacc = data.qacc_smooth
    if not model.opt.disableflags & _solver.DisableBit.WARMSTART:
        warm = _solver.Context.create(
            model, data.replace(qacc=data.qacc_warmstart), grad=False
        )
        smooth = _solver.Context.create(
            model, data.replace(qacc=data.qacc_smooth), grad=False
        )
        qacc = _solver.jp.where(
            warm.cost < smooth.cost,
            data.qacc_warmstart,
            data.qacc_smooth,
        )
    data = data.replace(qacc=qacc)

    ctx = _solver.Context.create(model, data)
    if model.opt.iterations == 1:
        ctx = body(ctx)
    else:
        ctx = _solver._while_loop_scan(
            cond,
            body,
            ctx,
            model.opt.iterations,
        )

    return data.tree_replace(
        {
            "qfrc_constraint": ctx.qfrc_constraint,
            "qacc": ctx.qacc,
            "_impl.efc_force": ctx.efc_force,
        }
    )


@contextmanager
def fixed_mjx_solver_outer_loop(*, semantic: str = CONVERGENCE_SCAN):
    """Temporarily install one explicit reverse-mode-safe solver semantic."""
    global _ACTIVE_SEMANTIC
    if semantic not in _SUPPORTED_SEMANTICS:
        raise ValueError(f"unsupported solver gradient semantic: {semantic}")
    if _ACTIVE_SEMANTIC is not None:
        raise RuntimeError("MJX solver gradient context is already active")
    original = _solver.solve
    original_loop = _solver._while_loop_scan
    _solver.solve = _solve_with_fixed_outer_loop
    if semantic == FIXED_SOLVER_AND_LINESEARCH_SCAN:
        _solver._while_loop_scan = _fixed_loop_scan
    _ACTIVE_SEMANTIC = semantic
    try:
        yield
    finally:
        _solver.solve = original
        _solver._while_loop_scan = original_loop
        _ACTIVE_SEMANTIC = None
