"""A process-local reverse-mode-safe outer loop for the MJX solver."""

from contextlib import contextmanager

from mujoco.mjx._src import solver as _solver


def _solve_with_fixed_outer_loop(model, data):
    """Match stock MJX solve while expressing its bounded outer loop as scan."""
    if not isinstance(model.opt._impl, _solver.OptionJAX):
        raise ValueError("solve requires JAX backend implementation")

    def cond(ctx):
        improvement = _solver._rescale(
            model, ctx.prev_cost - ctx.cost
        )
        gradient = _solver._rescale(
            model, _solver.math.norm(ctx.grad)
        )
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
            beta = _solver.jp.dot(
                ctx.grad, ctx.Mgrad - prev_mgrad
            )
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
def fixed_mjx_solver_outer_loop():
    """Temporarily replace only MJX solve; never patch global JAX control flow."""
    original = _solver.solve
    _solver.solve = _solve_with_fixed_outer_loop
    try:
        yield
    finally:
        _solver.solve = original
